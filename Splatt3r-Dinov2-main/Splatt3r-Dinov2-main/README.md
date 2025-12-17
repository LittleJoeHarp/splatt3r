# DINOv2 Integration in Splatt3R 


This project integrates **DINOv2** (Meta's self-supervised vision transformer) as a semantic feature backbone into the **Splatt3R** 3D Gaussian splatting pipeline. This is a part of integrating multiple feature backbones into an existing Splatt3R implementation like DINOv2,Croco MASt3R. The objective is to benchmark how different vision foundation models impact 3D scene reconstruction quality and efficiency.

## Project Objectives

1. **Enhance Feature Quality**: Augment MASt3R's geometry features with DINOv2's rich semantic representations
2. **Benchmark Backbones**: Compare DINOv2,Croco MASt3R within a Splatt3R-like pipeline
3. Analyze performance, speed, and memory across different model sizes
4. Test on datasets (CO3D) for broad applicability


### Core Additions

| Component | File | Purpose |
|-----------|------|---------|
| **DINOv2 Loader** | `src/model/splatt3r.py` | Load & manage DINOv2 models (vits14, vitb14, vitl14) |
| **Feature Fusion** | `src/model/feature_fusion.py` | Combine DINOv2 + MASt3R features (3 fusion strategies) |
| **Training Pipeline** | `main.py` | Integrate DINOv2 into MAST3RGaussians training loop |
| **Config System** | `configs/dinov2_co3d_experiment.yaml` | DINOv2 hyperparameter tuning |
| **Testing** | `test_forward.py` | Validate DINOv2 forward pass |

### Feature Fusion Strategies

**Three fusion modes supported:**

1. **Concatenation** (Default)
   - Combines MASt3R (768) + DINOv2_projected (768) → final (768)
   - Learnable projection reduces concatenated features back to 768 dims
   - Preserves complementary information from both models

2. **Addition**
   - Element-wise fusion: MASt3R + DINOv2_proj
   - Parameter-efficient, maintains spatial resolution
   - Requires dimension matching

3. **Weighted Sum**
   - Learnable weights: α·DINOv2 + (1-α)·MASt3R
   - Network learns optimal contribution ratio
   - Most flexible but adds parameters

## Technical Implementation

### Pipeline Architecture

```
Input Image Pair
    ↓
┌─────────────────────────────────────┐
│                                     │
├→ MASt3R Encoder (1024 dims)         │
│                                     │
├→ DINOv2 Backbone (384/768/1024)     │
│                                     │
├→ Feature Fusion Adapter             │
│                                     │
├→ MASt3R Decoder (768 dims)          │
│                                     │
├→ Gaussian Head                      │
│                                     │
└─────────────────────────────────────┘
    ↓
Novel View Synthesis & Rendering
```

### DINOv2 Variants Supported

| Model | Embed Dim | Speed | Quality | GPU Memory |
|-------|-----------|-------|---------|-----------|
| dinov2_vits14 | 384 | Fast ⚡ | Good | 2-4GB |
| dinov2_vitb14 | 768 | Medium ⚡⚡ | Better | 5-8GB |
| dinov2_vitl14 | 1024 | Slow ⚡⚡⚡ | Excellent | 10-15GB |



## Files Modified & Added

### New/Modified Files
-  `src/model/splatt3r.py` - DINOv2 integration core
-  `src/model/feature_fusion.py` - Feature fusion adapter
-  `main.py` - Training loop integration
-  `configs/dinov2_co3d_fast.yaml` - DINOv2 config
-  `test_forward.py` - Forward pass test

### Unchanged Files (from original Splatt3R)
- `utils/geometry.py`
- `utils/export.py`
- `utils/compute_ssim.py`
- `utils/loss_mask.py`
- `utils/sh_utils.py`


## Training Strategies

### Feature Extraction Only (Recommended)
- Freeze MASt3R + DINOv2
- Train only: projection layers, fusion adapter
- **Best for**: Fair backbone comparison


## Usage Quick Start


```powershell
python3 main.py configs/dinov2_co3d_experiment.yaml

```

## Dependencies

**Core Requirements:**
```
torch >= 2.0.0
torchvision >= 0.15.0
lightning >= 2.0.0
lpips >= 0.1.4
timm >= 0.9.0
einops >= 0.7.0
wandb >= 0.14.0
```

Install: `pip install -r requirements.txt`

---




