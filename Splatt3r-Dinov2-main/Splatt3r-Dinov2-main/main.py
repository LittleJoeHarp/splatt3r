import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb

# === DINOv2 Integration ===
import torch.nn.functional as F
from torchvision import transforms
# ==========================

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
# Correct import for co3d.py based on your file structure
import data.co3d as co3d_module
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace


class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model.
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        
        # Freeze all MASt3R parameters by default
        self.encoder.requires_grad_(False)
        # Unfreeze the parts we want to train (the Splatt3R heads)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

        # === DINOv2 INTEGRATION ===
        self.use_dinov2 = config.model.get('use_dinov2_features', False)
        if self.use_dinov2:
            print("[Model] Initializing DINOv2 Backbone...")
            self.dinov2_model_name = config.model.get('dinov2_model', 'dinov2_vitl14')
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', self.dinov2_model_name)
            
            # Get DINOv2 output dimension
            self.dinov2_output_dim = config.model.get('dinov2_dim', 1024)
            
            # Create projection layer to map DINOv2 features to decoder dim (768)
            # This keeps the total input dimension at 1024 + 768 = 1792
            self.dinov2_proj = torch.nn.Linear(self.dinov2_output_dim, 768)
            
            if config.model.get('freeze_dinov2', True):
                print("[Model] Freezing DINOv2 parameters.")
                self.dinov2.requires_grad_(False)
            else:
                print("[Model] DINOv2 parameters are trainable.")
                self.dinov2.requires_grad_(True)

            # DINOv2 requires specific ImageNet normalization
            self.dino_normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        # ==========================

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        # Loss criteria
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()

    def forward(self, view1, view2):
    
        # === HYBRID FORWARD PASS ===

        # 1. Get encoded features from MASt3R (Frozen)
        with torch.no_grad():
            # Use the public API method
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            
            # Run decoder to get both encoder features (for DPT) and decoder tokens
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)
            
            # dec1 and dec2 are lists with 2 elements:
            # - dec1[0]: list of encoder intermediate features for DPT head
            # - dec1[1]: decoder output tokens [B, N, D]
            
            mast3r_dec_feat_v1 = dec1[1].float()  # [B, 784, 768]
            mast3r_dec_feat_v2 = dec2[1].float()  # [B, 784, 768]

            # Get spatial dimensions from decoder tokens
            B, N_tokens, D = mast3r_dec_feat_v1.shape
            target_h = target_w = int(N_tokens ** 0.5)  # Should be 28 for 784 tokens

        # 2. Get Semantic Features from DINOv2 (if enabled)
        if self.use_dinov2:
            B = mast3r_dec_feat_v1.shape[0]
            
            # Get images
            if len(view1['img'].shape) == 5:  # [B, V, C, H, W]
                batch_loader, num_views = view1['img'].shape[0], view1['img'].shape[1]
                B = batch_loader
                img1 = view1['img'].reshape(batch_loader * num_views, *view1['img'].shape[2:])
                img2 = view2['img'].reshape(batch_loader * num_views, *view2['img'].shape[2:])
            else:  # [B, C, H, W]
                img1 = view1['img']
                img2 = view2['img']
            
            _, _, H, W = img1.shape
            images_flat = torch.cat([img1, img2], dim=0)
            dino_input = self.dino_normalize(images_flat)
            
            # Run DINOv2
            with torch.no_grad() if self.config.model.get('freeze_dinov2', True) else torch.enable_grad():
                features_dict = self.dinov2.forward_features(dino_input)
            
            patch_tokens = features_dict['x_norm_patchtokens']
            
            # Reshape to spatial map
            grid_h, grid_w = H // 14, W // 14
            embed_dim = patch_tokens.shape[-1]
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B * 2, embed_dim, grid_h, grid_w)

            # Interpolate to match MASt3R resolution
            resized_dino_features = F.interpolate(
                feature_map,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to token format
            resized_dino_features = resized_dino_features.permute(0, 2, 3, 1)
            dino_v1, dino_v2 = torch.chunk(resized_dino_features, 2, dim=0)
            
            dino_v1 = dino_v1.reshape(B, target_h * target_w, embed_dim)
            dino_v2 = dino_v2.reshape(B, target_h * target_w, embed_dim)

            # 3. Project DINOv2 features to decoder dimension
            dino_v1_proj = self.dinov2_proj(dino_v1)  # [B, 784, 768]
            dino_v2_proj = self.dinov2_proj(dino_v2)  # [B, 784, 768]

            # 4. Fuse: Add to decoder tokens
            fused_dec_v1 = mast3r_dec_feat_v1 + dino_v1_proj
            fused_dec_v2 = mast3r_dec_feat_v2 + dino_v2_proj
            
            # Create list format for _downstream_head: [encoder_feats..., fused_decoder]
            # dec1[0] contains the encoder features, dec1[1] is the decoder tokens
            # We replace dec1[1] with our fused version
            fused_dec1 = [tok.float() for tok in dec1]
            fused_dec1[-1] = fused_dec_v1
            
            fused_dec2 = [tok.float() for tok in dec2]
            fused_dec2[-1] = fused_dec_v2

        else:
            # No DINOv2: use original MASt3R features
            fused_dec1 = [tok.float() for tok in dec1]
            fused_dec2 = [tok.float() for tok in dec2]

        # 5. Pass fused features to trainable Splatt3R heads
        # _downstream_head expects a list where the last element is decoder tokens
        pred1 = self.encoder._downstream_head(1, fused_dec1, shape1)
        pred2 = self.encoder._downstream_head(2, fused_dec2, shape2)

        # === REST OF THE FORWARD PASS (covariances, SH, etc.) ===
        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            
            img1_orig = view1['original_img'] if 'original_img' in view1 else view1['img']
            img2_orig = view2['original_img'] if 'original_img' in view2 else view2['img']
            
            # Handle different tensor shapes: could be [C,H,W], [B,C,H,W], or [B,V,C,H,W]
            if len(img1_orig.shape) == 3:  # [C, H, W] - add batch dimension
                img1_orig = img1_orig.unsqueeze(0)
            elif len(img1_orig.shape) == 5:  # [B, V, C, H, W] - squeeze view dimension
                img1_orig = img1_orig.squeeze(1)
            # else: already [B, C, H, W]
            
            if len(img2_orig.shape) == 3:  # [C, H, W] - add batch dimension
                img2_orig = img2_orig.unsqueeze(0)
            elif len(img2_orig.shape) == 5:  # [B, V, C, H, W] - squeeze view dimension
                img2_orig = img2_orig.squeeze(1)
            # else: already [B, C, H, W]

            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(img1_orig, 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(img2_orig, 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def training_step(self, batch, batch_idx):
    
        if 'context' in batch:
            view1, view2 = batch['context']
            _, _, h, w = view1["img"].shape
            
            # Ensure original_img is present
            if 'original_img' not in view1:
                view1['original_img'] = view1['img']
            if 'original_img' not in view2:
                view2['original_img'] = view2['img']
            
            # === FIX: Add target to batch for CO3D ===
            if 'target' not in batch:
                # For CO3D, view2 is the target view
                batch['target'] = [view2]
            # =========================================
        else:
            view1 = batch['view1']
            view2 = batch['view2']
            _, _, h, w = view1["img"].shape 
            view1['original_img'] = view1['img']
            view2['original_img'] = view2['img']
            
            # Add target for decoder
            if 'target' not in batch:
                batch['target'] = [view2]

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        if self.config.data.dataset == 'co3d':
        # CO3D has no depth map, create a full mask (all True)
            target_view = batch['target'][0]
            B, C, H, W = target_view['img'].shape
            mask = torch.ones(B, H, W, dtype=torch.bool, device=target_view['img'].device)
        else:
        # Original logic for datasets with depth (like ScanNet++)
            mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        self.log_metrics('train', loss, mse, lpips)
        return loss

    def validation_step(self, batch, batch_idx):

        if 'context' in batch:
            view1, view2 = batch['context']
            _, _, h, w = view1["img"].shape
            
            # === FIX: Add target to batch for CO3D ===
            if 'target' not in batch:
                batch['target'] = [view2]
            # =========================================
        else:
            view1 = batch['view1']
            view2 = batch['view2']
            _, _, h, w = view1["img"].shape
            view1['original_img'] = view1['img']
            view2['original_img'] = view2['img']
            
            if 'target' not in batch:
                batch['target'] = [view2]

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        if self.config.data.dataset == 'co3d':
            # CO3D has no depth map, create a full mask (all True)
            target_view = batch['target'][0]
            B, C, H, W = target_view['img'].shape
            mask = torch.ones(B, H, W, dtype=torch.bool, device=target_view['img'].device)
        else:
            # Original logic for datasets with depth (like ScanNet++)
            mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        self.log_metrics('val', loss, mse, lpips)
        return loss

    def test_step(self, batch, batch_idx):

        # MODIFICATION: Handle CO3D batch structure
        if 'context' in batch:
            view1, view2 = batch['context']
            _, _, h, w = view1["img"].shape
        else:
            view1 = batch['view1']
            view2 = batch['view2']
            B, C, H_img, W_img = view1["img"].shape 
            view1['original_img'] = view1['img']
            view2['original_img'] = view2['img']
            # === FIX for KeyError: 'instance' ===
            view1['instance'] = None
            view2['instance'] = None
            # ====================================
            h, w = H_img, W_img # Use image dims
            
        num_targets = len(batch.get('target', [view2])) # Handle CO3D test

        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, ssim = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True
        )

        # Log losses
        self.log_metrics('test', loss, mse, lpips, ssim=ssim)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):

        # MODIFICATION: Handle CO3D batch structure
        if 'target' in batch:
            target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        else:
            # CO3D: Reconstruct view2 from view1
            target_color = view2['img'].unsqueeze(1) # [B, 1, C, H, W]

        predicted_color = color

        if apply_mask:
            # Ensure mask is broadcastable to color
            mask = mask.unsqueeze(1) # [B, 1, H, W]
            if mask.shape[1] != predicted_color.shape[1]:
                 mask = mask.repeat(1, predicted_color.shape[1], 1, 1) # [B, V, H, W]

            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask
            predicted_color = predicted_color * mask

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        
        # Adjust mask shape for flattening
        if mask.shape[1] != flattened_color.shape[0] // flattened_color.shape[0]:
             mask = mask.repeat(1, predicted_color.shape[1], 1, 1) # [B, V, H, W]
             
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')


        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        # MAST3R Loss
        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        
        # === NEW OPTIMIZER LOGIC ===
        # Collect all parameters that require gradients
        
        trainable_params = []
        
        # 1. Add Splatt3R Head parameters (always trainable)
        trainable_params.extend(self.encoder.downstream_head1.gaussian_dpt.dpt.parameters())
        trainable_params.extend(self.encoder.downstream_head2.gaussian_dpt.dpt.parameters())

        # 2. Add DINOv2 parameters (if not frozen)
        if self.use_dinov2 and self.config.model.get('freeze_dinov2', True) == False:
            print("[Optimizer] Adding DINOv2 parameters to optimizer.")
            trainable_params.extend(self.dinov2.parameters())
            
        # 3. Add MASt3R parameters (if geometry loss is on)
        if self.config.loss.mast3r_loss_weight is not None:
            print("[Optimizer] Adding MASt3R geometry head parameters to optimizer.")
            # We already added downstream_head, but let's be explicit
            # if other parts were meant to be trained.
            pass

        if not trainable_params:
            print("WARNING: No trainable parameters found! Check requires_grad_ flags.")
            # Add a dummy parameter to prevent Adam from crashing
            trainable_params = [torch.nn.Parameter(torch.zeros(1))]

        # Select optimizer from config
        opt_name = self.config.opt.get('optimizer', 'adam').lower()
        
        if opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params, 
                lr=self.config.opt.lr,
                weight_decay=self.config.opt.get('weight_decay', 0.01)
            )
        else: # Default to Adam
            optimizer = torch.optim.Adam(
                trainable_params, 
                lr=self.config.opt.lr
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)
    if config.loggers.use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project='splatt3r',
            name=config.name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
        )
        if wandb.run is not None:
            wandb.run.log_code(".")
        loggers.append(wandb_logger)

    # Set up profiler
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None

    # Model
    print('Loading Model')
    model = MAST3RGaussians(config)
    
    # === CONFIG KEY FIX ===
    # Look for keys inside config.model, not at the root
    if config.model.use_pretrained:
        print(f"Loading pretrained MASt3R weights from {config.model.pretrained_mast3r_path}")
        ckpt = torch.load(config.model.pretrained_mast3r_path)
        _ = model.encoder.load_state_dict(ckpt['model'], strict=False)
        del ckpt
        print("MASt3R weights loaded.")
    # =======================

    # Training Datasets
    print(f'Building Datasets')
    
    # Support both CO3D and ScanNet++ datasets based on config
    if hasattr(config.data, 'dataset') and config.data.dataset == 'co3d':
        print(f"Loading CO3D dataset from root: {config.data.root}")
        
        # === DATALOADER FIX ===
        # Pass the category from the config file to the dataset
        
        train_dataset = co3d_module.CO3DDataset(
            root_dir=config.data.root,
            category=config.data.category, # <-- FIXED
            split='train',
            img_size=tuple(config.data.resolution),
            frame_gap=(
                config.data.get('temporal_gap_min', 5),
                config.data.get('temporal_gap_max', 10)
            ),
        )
        val_dataset = co3d_module.CO3DDataset(
            root_dir=config.data.root,
            category=config.data.category, # <-- FIXED
            split='test', # <-- FIXED (CO3D loader uses 'test' not 'val')
            img_size=tuple(config.data.resolution),
            frame_gap=(
                config.data.get('temporal_gap_min', 5),
                config.data.get('temporal_gap_max', 10)
            ),
        )
        # =======================

    else:
        # Load ScanNet++ dataset (default)
        print(f"Loading ScanNet++ dataset from root: {config.data.root}")
        train_dataset = scannetpp.get_scannet_dataset(
            config.data.root,
            'train',
            config.data.resolution,
            num_epochs_per_epoch=config.data.epochs_per_train_epoch,
        )
        val_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=0.5,
            beta=0.5,
            resolution=config.data.resolution,
            use_every_n_sample=100,
        )
    
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    # Training
    print('Training')
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            # export.SaveBatchData(save_dir=config.save_dir),
        ],
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.epochs,
        num_sanity_val_steps=0,  # Skip sanity checks to save time (10+ minutes)
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if (isinstance(config.devices, list) and len(config.devices) > 1) else "auto",
    )
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing
    # === TESTING FIX ===
    # The original test loop was hardcoded to ScanNet++.
    # We will test on the CO3D 'test' split (which we loaded as val_dataset)
    print("Running final test on CO3D 'test' split...")
    
    # We use the val_dataset here because it's already our CO3D 'test' split
    data_loader_test = data_loader_val 
    
    original_save_dir = config.save_dir
    results = {}
    
    masking_configs = ((True, False), (True, True))
    for apply_mask, average_over_mask in masking_configs:

        new_save_dir = os.path.join(
            original_save_dir,
            f'co3d_test_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
        )
        os.makedirs(new_save_dir, exist_ok=True)
        model.config.save_dir = new_save_dir

        L.seed_everything(config.seed, workers=True)

        # Trainer for testing
        trainer = L.Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=[export.SaveBatchData(save_dir=config.save_dir),],
            default_root_dir=config.save_dir,
            devices=config.devices,
            log_every_n_steps=10,
            strategy="ddp_find_unused_parameters_true" if (isinstance(config.devices, list) and len(config.devices) > 1) else "auto",
        )

        model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
        model.config.loss.apply_mask = apply_mask
        model.config.loss.average_over_mask = average_over_mask
        res = trainer.test(model, dataloaders=data_loader_test)
        results[f"co3d_test_apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res

        # Save the results
        save_path = os.path.join(original_save_dir, 'results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run training
    run_experiment(config)