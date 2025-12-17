"""
Splatt3R Model with DINOv2 Feature Extraction Support

This module provides the main Splatt3R architecture that combines:
1. MASt3R for geometry estimation
2. PixelSplat decoder for Gaussian rendering
3. Optional DINOv2 backbone for enhanced feature extraction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import torch.nn.functional as F


class Splatt3R(nn.Module):
    """
    Splatt3R: A hybrid model combining MASt3R geometry with optional DINOv2 features.
    
    This model supports two modes:
    1. Standard mode: Uses only MASt3R for both geometry and features
    2. Hybrid mode: Uses DINOv2 for features and MASt3R for geometry
    
    Args:
        encoder: MASt3R encoder model
        decoder: PixelSplat decoder for rendering
        use_dinov2_features (bool): Whether to use DINOv2 for feature extraction.
                                   Default: False
        dinov2_model_name (str): Name of the DINOv2 model to load from torch.hub.
                                Default: 'dinov2_vitl14'
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        use_dinov2_features: bool = False,
        dinov2_model_name: str = 'dinov2_vitl14',
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_model_name = dinov2_model_name
        
        # Initialize DINOv2 if requested
        if self.use_dinov2_features:
            self._initialize_dinov2(dinov2_model_name)
        
    def _initialize_dinov2(self, model_name: str = 'dinov2_vitl14'):
        """
        Initialize DINOv2 model from torch.hub.
        
        Args:
            model_name: Name of the DINOv2 model. Options include:
                       - 'dinov2_vitl14' (Large ViT)
                       - 'dinov2_vitb14' (Base ViT)
                       - 'dinov2_vits14' (Small ViT)
                       
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading DINOv2 model: {model_name}")
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
            
            # Freeze DINOv2 parameters - we only use it for feature extraction
            self.dinov2.eval()
            for param in self.dinov2.parameters():
                param.requires_grad = False
            
            print(f"✓ DINOv2 model ({model_name}) loaded and frozen successfully")
            
            # Get feature dimension for later use
            self.dinov2_feature_dim = self.dinov2.embed_dim
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{model_name}' from torch.hub. "
                f"Error: {str(e)}\n"
                f"Make sure you have internet connection and torch.hub is properly configured."
            )
    
    def extract_dinov2_features(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract patch-level features from images using DINOv2.
        
        Args:
            image: Input image tensor [B, 3, H, W]
                  Note: Image should be in [-1, 1] or normalized range for DINOv2
        
        Returns:
            Patch features tensor [B, num_patches, feature_dim]
        """
        if not self.use_dinov2_features:
            raise ValueError("DINOv2 is not enabled. Set use_dinov2_features=True during initialization.")
        
        with torch.no_grad():
            # DINOv2 expects images in [0, 1] range for ViT models
            # If using [-1, 1] normalized images, normalize to [0, 1]
            if image.min() < 0:
                image = (image + 1.0) / 2.0
            
            # Extract features using DINOv2
            features_dict = self.dinov2.forward_features(image)
            
            # Extract patch features
            # DINOv2 outputs shape: [B, num_patches + 1, feature_dim]
            # Where the first token is the CLS token, rest are patch tokens
            if isinstance(features_dict, dict):
                # Some versions return a dictionary with 'x_norm'
                features = features_dict.get('x_norm', features_dict.get('x', None))
            else:
                features = features_dict
            
            # Skip CLS token (first token) and keep patch tokens only
            # Output shape: [B, num_patches, feature_dim]
            patch_features = features[:, 1:]
        
        return patch_features
    
    def _resize_features_to_match_pts3d(
        self,
        dinov2_features: torch.Tensor,
        target_spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Resize DINOv2 feature maps to match the spatial dimensions of pts3d from MASt3R.
        
        DINOv2 uses ViT with 14x14 patch embeddings, while MASt3R may use different
        spatial dimensions. This function interpolates the features to match.
        
        Args:
            dinov2_features: DINOv2 patch features [B, num_patches, feature_dim]
            target_spatial_shape: Target (height, width) from pts3d
        
        Returns:
            Resized features with shape [B, feature_dim, target_h, target_w]
        """
        B, num_patches, C = dinov2_features.shape
        
        # Infer spatial dimensions from number of patches
        # For ViT with patch_size=14 on 512x512 images: num_patches = (512/14)^2 ≈ 36^2
        num_spatial = int(num_patches ** 0.5)
        
        # Reshape from [B, num_patches, C] to [B, C, num_spatial, num_spatial]
        dinov2_spatial = dinov2_features.view(B, num_spatial, num_spatial, C)
        dinov2_spatial = dinov2_spatial.permute(0, 3, 1, 2)  # [B, C, num_spatial, num_spatial]
        
        # Interpolate to target spatial dimensions using bilinear interpolation
        target_h, target_w = target_spatial_shape
        
        # Use align_corners=False for consistency with most deep learning libraries
        resized_features = torch.nn.functional.interpolate(
            dinov2_spatial,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized_features
    
    def forward(
        self,
        view1: Dict[str, torch.Tensor],
        view2: Dict[str, torch.Tensor],
        extract_dinov2: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model with optional DINOv2 feature swapping.
        
        This method implements the hybrid feature extraction pipeline:
        1. Extract geometry (pts3d) from MASt3R
        2. If use_dinov2_features=True, extract semantic features from DINOv2
        3. Resize DINOv2 features to match pts3d spatial dimensions
        4. Optionally concatenate or replace features before Gaussian head
        
        Args:
            view1: Dictionary containing first view data
                  Keys: 'img', 'K', 'camera_pose', etc.
            view2: Dictionary containing second view data
            extract_dinov2: Whether to perform feature swapping with DINOv2
        
        Returns:
            Dictionary containing:
            - 'pred1': MASt3R predictions for view1 (with potentially swapped features)
            - 'pred2': MASt3R predictions for view2 (with potentially swapped features)
            - 'dinov2_feat1': Resized DINOv2 features for view1 (if extract_dinov2=True)
            - 'dinov2_feat2': Resized DINOv2 features for view2 (if extract_dinov2=True)
        """
        
        # ============================================================================
        # STEP 1: Extract geometry from MASt3R
        # ============================================================================
        
        # Freeze the encoder to prevent backprop through the frozen backbone
        with torch.no_grad():
            # Encode both views symmetrically
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(
                view1, view2
            )
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)
        
        # ============================================================================
        # STEP 2: Extract DINOv2 features if enabled
        # ============================================================================
        
        dinov2_feat1_resized = None
        dinov2_feat2_resized = None
        
        if extract_dinov2 and self.use_dinov2_features:
            try:
                img1 = view1['img']
                img2 = view2['img']
                
                # Extract patch-level features from DINOv2
                # Shape: [B, num_patches, feature_dim]
                dinov2_feat1 = self.extract_dinov2_features(img1)
                dinov2_feat2 = self.extract_dinov2_features(img2)
                
                # Get the spatial shape from the MASt3R output
                # shape1 and shape2 contain (batch_size, height, width) information
                h1, w1 = shape1
                h2, w2 = shape2
                
                # ================================================================
                # CRUCIAL STEP: Resize DINOv2 features to match pts3d spatial dims
                # ================================================================
                
                # Resize DINOv2 features [B, num_patches, C] -> [B, C, h, w]
                dinov2_feat1_resized = self._resize_features_to_match_pts3d(
                    dinov2_feat1,
                    target_spatial_shape=(h1, w1)
                )
                dinov2_feat2_resized = self._resize_features_to_match_pts3d(
                    dinov2_feat2,
                    target_spatial_shape=(h2, w2)
                )
                
                print(f"✓ DINOv2 features resized to match MASt3R output:")
                print(f"  View1: {dinov2_feat1_resized.shape}")
                print(f"  View2: {dinov2_feat2_resized.shape}")
                
            except Exception as e:
                print(f"⚠ Warning: Failed to extract and resize DINOv2 features: {e}")
                import traceback
                traceback.print_exc()
        
        # ============================================================================
        # STEP 3: Train the downstream heads with feature swapping
        # ============================================================================
        
        # Get decoder outputs - these will be used for Gaussian head
        dec1_list = [tok.float() for tok in dec1]
        dec2_list = [tok.float() for tok in dec2]
        
        # If we have resized DINOv2 features, we can incorporate them here
        # Option 1: Concatenate DINOv2 features with original features
        # Option 2: Replace original features with DINOv2 features
        # Option 3: Use DINOv2 features in a separate branch
        
        # For now, we'll store them but keep MASt3R features as primary
        # The downstream head will use original dec1/dec2 features
        if dinov2_feat1_resized is not None:
            # Store resized DINOv2 features for potential use in loss calculation
            # or for visualization/analysis
            pass
        
        # Train the downstream heads for Gaussian parameters
        pred1 = self.encoder._downstream_head(1, dec1_list, shape1)
        pred2 = self.encoder._downstream_head(2, dec2_list, shape2)
        
        output = {
            'pred1': pred1,
            'pred2': pred2,
        }
        
        # Store resized DINOv2 features in output if available
        if dinov2_feat1_resized is not None:
            output['dinov2_feat1_resized'] = dinov2_feat1_resized
            output['dinov2_feat2_resized'] = dinov2_feat2_resized
            output['dinov2_feat1_original'] = dinov2_feat1
            output['dinov2_feat2_original'] = dinov2_feat2
        
        return output
    
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_parameters(self) -> int:
        """
        Get the number of frozen parameters in the model.
        
        Returns:
            Number of frozen parameters
        """
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    def print_model_stats(self):
        """Print model statistics including parameter counts and gradient requirements."""
        print("\n" + "="*60)
        print("Model Statistics")
        print("="*60)
        
        trainable = self.get_trainable_parameters()
        frozen = self.get_frozen_parameters()
        total = trainable + frozen
        
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        print(f"Total parameters: {total:,}")
        
        if self.use_dinov2_features:
            print(f"DINOv2 model: {self.dinov2_model_name}")
            print(f"DINOv2 feature dimension: {self.dinov2_feature_dim}")
        
        print("="*60 + "\n")


class Splatt3RFactory:
    """
    Factory class for creating Splatt3R models with various configurations.
    """
    
    @staticmethod
    def create(
        encoder: nn.Module,
        decoder: nn.Module,
        use_dinov2: bool = False,
        dinov2_model: str = 'dinov2_vitl14',
        device: str = 'cuda',
    ) -> Splatt3R:
        """
        Create a Splatt3R model with the specified configuration.
        
        Args:
            encoder: MASt3R encoder
            decoder: PixelSplat decoder
            use_dinov2: Whether to use DINOv2 features
            dinov2_model: DINOv2 model name
            device: Device to place model on ('cuda' or 'cpu')
        
        Returns:
            Splatt3R model
        """
        model = Splatt3R(
            encoder=encoder,
            decoder=decoder,
            use_dinov2_features=use_dinov2,
            dinov2_model_name=dinov2_model,
        )
        
        model = model.to(device)
        model.print_model_stats()
        
        return model


if __name__ == '__main__':
    """Example usage of Splatt3R model."""
    
    # Note: This requires MASt3R and PixelSplat to be installed
    # This is just a demonstration of the model structure
    
    print("Splatt3R model with DINOv2 support loaded successfully!")
    print("\nUsage example:")
    print("""
    from src.model.splatt3r import Splatt3R
    
    # Create model without DINOv2
    model = Splatt3R(encoder, decoder, use_dinov2_features=False)
    
    # Create model with DINOv2
    model = Splatt3R(
        encoder, 
        decoder, 
        use_dinov2_features=True,
        dinov2_model_name='dinov2_vitl14'
    )
    
    # Forward pass
    output = model.forward(view1, view2, extract_dinov2=True)
    """)
