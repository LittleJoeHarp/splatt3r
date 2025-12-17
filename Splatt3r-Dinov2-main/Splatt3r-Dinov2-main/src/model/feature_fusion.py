"""
Feature Fusion Adapter for DINOv2 + MASt3R Integration

This module provides adapters to fuse DINOv2 semantic features with MASt3R
geometry features, handling dimension mismatches and feature concatenation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class DINOv2FeatureFusionAdapter(nn.Module):
    """
    Adapter module for fusing DINOv2 features with MASt3R decoder features.
    
    DINOv2 (vitl14) outputs 1024 channels while MASt3R outputs typically 768 channels.
    This adapter handles:
    1. Optional projection of DINOv2 features to match decoder dim
    2. Concatenation or addition of features
    3. Proper layer normalization
    
    Args:
        dinov2_dim (int): Input dimension from DINOv2. Default: 1024
        mast3r_dim (int): Input dimension from MASt3R decoder. Default: 768
        output_dim (int, optional): Output dimension. If None, uses mast3r_dim
        fusion_mode (str): How to fuse features. Options:
            - 'concat': Concatenate features [mast3r_dim + dinov2_proj_dim]
            - 'add': Element-wise addition (requires same dims)
            - 'weighted_sum': Learnable weighted sum (alpha * mast3r + (1-alpha) * dinov2)
            Default: 'concat'
    """
    
    def __init__(
        self,
        dinov2_dim: int = 1024,
        mast3r_dim: int = 768,
        output_dim: Optional[int] = None,
        fusion_mode: str = 'concat',
    ):
        super().__init__()
        
        self.dinov2_dim = dinov2_dim
        self.mast3r_dim = mast3r_dim
        self.fusion_mode = fusion_mode
        
        # If no output dim specified, use mast3r_dim
        if output_dim is None:
            output_dim = mast3r_dim
        self.output_dim = output_dim
        
        # Project DINOv2 features to match output dimension
        if fusion_mode == 'concat':
            # Project DINOv2 to output_dim, then concatenate
            self.dinov2_proj = nn.Sequential(
                nn.Conv2d(dinov2_dim, output_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=output_dim),
            )
            # Project MASt3R features to output_dim
            self.mast3r_proj = nn.Sequential(
                nn.Conv2d(mast3r_dim, output_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=output_dim),
            )
            # Output projection to reduce concatenated dims
            concat_dim = 2 * output_dim
            self.output_proj = nn.Sequential(
                nn.Conv2d(concat_dim, output_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=output_dim),
                nn.ReLU(inplace=True),
            )
        
        elif fusion_mode == 'add':
            # For addition mode, project both to same dimension
            if dinov2_dim != mast3r_dim:
                self.dinov2_proj = nn.Sequential(
                    nn.Conv2d(dinov2_dim, output_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=32, num_channels=output_dim),
                )
            else:
                self.dinov2_proj = nn.Identity()
            
            if mast3r_dim != output_dim:
                self.mast3r_proj = nn.Sequential(
                    nn.Conv2d(mast3r_dim, output_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=32, num_channels=output_dim),
                )
            else:
                self.mast3r_proj = nn.Identity()
        
        elif fusion_mode == 'weighted_sum':
            # Project DINOv2 to mast3r_dim
            if dinov2_dim != output_dim:
                self.dinov2_proj = nn.Sequential(
                    nn.Conv2d(dinov2_dim, output_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=32, num_channels=output_dim),
                )
            else:
                self.dinov2_proj = nn.Identity()
            
            if mast3r_dim != output_dim:
                self.mast3r_proj = nn.Sequential(
                    nn.Conv2d(mast3r_dim, output_dim, kernel_size=1),
                    nn.GroupNorm(num_groups=32, num_channels=output_dim),
                )
            else:
                self.mast3r_proj = nn.Identity()
            
            # Learnable weight for DINOv2 features
            self.dinov2_weight = nn.Parameter(torch.tensor(0.5))
        
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        print(f"✓ DINOv2FeatureFusionAdapter initialized:")
        print(f"  DINOv2 input dim: {dinov2_dim}")
        print(f"  MASt3R input dim: {mast3r_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Fusion mode: {fusion_mode}")
    
    def forward(
        self,
        mast3r_features: torch.Tensor,
        dinov2_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse MASt3R and DINOv2 features.
        
        Args:
            mast3r_features: [B, C_mast3r, H, W]
            dinov2_features: [B, C_dinov2, H, W]
        
        Returns:
            Fused features [B, C_out, H, W]
        """
        
        if self.fusion_mode == 'concat':
            # Project both features
            mast3r_proj = self.mast3r_proj(mast3r_features)
            dinov2_proj = self.dinov2_proj(dinov2_features)
            
            # Concatenate
            fused = torch.cat([mast3r_proj, dinov2_proj], dim=1)
            
            # Project output
            output = self.output_proj(fused)
        
        elif self.fusion_mode == 'add':
            # Project and add
            mast3r_proj = self.mast3r_proj(mast3r_features)
            dinov2_proj = self.dinov2_proj(dinov2_features)
            output = mast3r_proj + dinov2_proj
        
        elif self.fusion_mode == 'weighted_sum':
            # Learnable weighted sum
            mast3r_proj = self.mast3r_proj(mast3r_features)
            dinov2_proj = self.dinov2_proj(dinov2_features)
            
            # Apply softmax to weight for numerical stability
            alpha = torch.sigmoid(self.dinov2_weight)
            output = alpha * dinov2_proj + (1 - alpha) * mast3r_proj
        
        return output


class DINOv2DecoderFeatureFusion(nn.Module):
    """
    Specialized fusion module for MASt3R decoder outputs with DINOv2 features.
    
    This handles the case where we want to inject DINOv2 semantic features
    directly into the decoder pipeline before the Gaussian head.
    
    Args:
        dec_embed_dim (int): MASt3R decoder embedding dimension. Default: 768
        dinov2_dim (int): DINOv2 output dimension. Default: 1024
        fusion_mode (str): Fusion strategy ('concat', 'add', 'weighted_sum'). Default: 'concat'
    """
    
    def __init__(
        self,
        dec_embed_dim: int = 768,
        dinov2_dim: int = 1024,
        fusion_mode: str = 'concat',
    ):
        super().__init__()
        
        self.dec_embed_dim = dec_embed_dim
        self.dinov2_dim = dinov2_dim
        self.fusion_mode = fusion_mode
        
        # Determine output dimension based on fusion mode
        if fusion_mode == 'concat':
            # Output will be processed then reduced back to dec_embed_dim
            output_dim = dec_embed_dim
        else:
            output_dim = dec_embed_dim
        
        # Create adapter
        self.adapter = DINOv2FeatureFusionAdapter(
            dinov2_dim=dinov2_dim,
            mast3r_dim=dec_embed_dim,
            output_dim=output_dim,
            fusion_mode=fusion_mode,
        )
    
    def forward(
        self,
        decoder_features: torch.Tensor,
        dinov2_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse decoder features with DINOv2 features.
        
        Args:
            decoder_features: [B, seq_len, dec_embed_dim] or [B, dec_embed_dim, H, W]
            dinov2_features: [B, dinov2_dim, H, W]
        
        Returns:
            Fused features with shape matching decoder_features
        """
        # Handle both sequence and spatial formats
        if decoder_features.ndim == 3:
            # Sequence format: [B, seq_len, C]
            B, seq_len, C = decoder_features.shape
            # Convert to spatial for fusion
            H = W = int(seq_len ** 0.5)
            decoder_spatial = decoder_features.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            decoder_spatial = decoder_features
        
        # Fuse features
        fused = self.adapter(decoder_spatial, dinov2_features)
        
        # Convert back to sequence format if needed
        if decoder_features.ndim == 3:
            B, C, H, W = fused.shape
            fused = fused.permute(0, 2, 3, 1).view(B, -1, C)
        
        return fused


class DINOv2ConfigurableInputHead(nn.Module):
    """
    Configurable input dimension handler for Gaussian Head.
    
    This module ensures that features can be concatenated with DINOv2 features
    while maintaining compatibility with the Gaussian head input requirements.
    
    The Gaussian head expects a concatenation of encoder and decoder outputs:
    input_dim = enc_embed_dim + dec_embed_dim
    
    If DINOv2 features are concatenated:
    input_dim = enc_embed_dim + dec_embed_dim + dinov2_dim
    
    Args:
        enc_embed_dim (int): Encoder embedding dimension. Default: 1024
        dec_embed_dim (int): Decoder embedding dimension. Default: 768
        dinov2_dim (int, optional): DINOv2 dimension if using features. Default: None
        use_dinov2 (bool): Whether to concatenate DINOv2 features. Default: False
    """
    
    def __init__(
        self,
        enc_embed_dim: int = 1024,
        dec_embed_dim: int = 768,
        dinov2_dim: Optional[int] = None,
        use_dinov2: bool = False,
    ):
        super().__init__()
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.dinov2_dim = dinov2_dim or 1024
        self.use_dinov2 = use_dinov2
        
        # Calculate total input dimension
        self.base_input_dim = enc_embed_dim + dec_embed_dim
        
        if use_dinov2:
            self.total_input_dim = self.base_input_dim + self.dinov2_dim
        else:
            self.total_input_dim = self.base_input_dim
        
        print(f"✓ DINOv2ConfigurableInputHead initialized:")
        print(f"  Encoder dim: {enc_embed_dim}")
        print(f"  Decoder dim: {dec_embed_dim}")
        print(f"  Base input dim (enc + dec): {self.base_input_dim}")
        if use_dinov2:
            print(f"  DINOv2 dim: {self.dinov2_dim}")
            print(f"  Total input dim (with DINOv2): {self.total_input_dim}")
        else:
            print(f"  DINOv2 disabled")
    
    def get_input_dim(self) -> int:
        """Get the total input dimension for the Gaussian head."""
        return self.total_input_dim
    
    def get_base_input_dim(self) -> int:
        """Get the base input dimension without DINOv2."""
        return self.base_input_dim


if __name__ == '__main__':
    """Example usage of feature fusion adapters."""
    
    print("="*60)
    print("DINOv2 Feature Fusion Adapters")
    print("="*60)
    
    # Example 1: Concatenation mode
    print("\n1. Concatenation Fusion:")
    adapter_concat = DINOv2FeatureFusionAdapter(
        dinov2_dim=1024,
        mast3r_dim=768,
        output_dim=768,
        fusion_mode='concat'
    )
    
    # Example 2: Addition mode
    print("\n2. Addition Fusion:")
    adapter_add = DINOv2FeatureFusionAdapter(
        dinov2_dim=1024,
        mast3r_dim=768,
        output_dim=768,
        fusion_mode='add'
    )
    
    # Example 3: Weighted sum mode
    print("\n3. Weighted Sum Fusion:")
    adapter_weighted = DINOv2FeatureFusionAdapter(
        dinov2_dim=1024,
        mast3r_dim=768,
        output_dim=768,
        fusion_mode='weighted_sum'
    )
    
    # Example 4: Configurable input head
    print("\n4. Configurable Input Head (with DINOv2):")
    config_head = DINOv2ConfigurableInputHead(
        enc_embed_dim=1024,
        dec_embed_dim=768,
        dinov2_dim=1024,
        use_dinov2=True
    )
    
    print("\n" + "="*60)
    print("All components initialized successfully!")
    print("="*60)
