#!/usr/bin/env python3
"""
Quick test to verify the forward pass works without full training
"""
import sys
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

import torch
import omegaconf
from main import MAST3RGaussians
import data.co3d as co3d_module
import workspace

print("Loading config...")
config = workspace.load_config("configs/dinov2_co3d_experiment.yaml", [])

print("Loading model...")
model = MAST3RGaussians(config)
model = model.cuda()
model.eval()

print("Loading dataset...")
dataset = co3d_module.CO3DDataset(
    root_dir=config.data.root,
    category=config.data.category,
    split='train',
    img_size=tuple(config.data.resolution),
    frame_gap=(5, 10),
)

print(f"Dataset size: {len(dataset)}")
print("Getting first sample...")
sample = dataset[0]

print(f"Sample keys: {sample.keys()}")
batch = {
    'context': sample['context'],
    'target': [sample['context'][1]],
    'scene': sample['scene']
}

# Move to CUDA
view1, view2 = batch['context']

# Properly batch the data
view1['img'] = view1['img'].unsqueeze(0).cuda()
view1['original_img'] = view1.get('original_img', view1['img']).unsqueeze(0).cuda() if view1.get('original_img') is not None else view1['img']
view1['camera_pose'] = view1['camera_pose'].unsqueeze(0).cuda()
view1['camera_intrinsics'] = view1['camera_intrinsics'].unsqueeze(0).cuda()

view2['img'] = view2['img'].unsqueeze(0).cuda()
view2['original_img'] = view2.get('original_img', view2['img']).unsqueeze(0).cuda() if view2.get('original_img') is not None else view2['img']
view2['camera_pose'] = view2['camera_pose'].unsqueeze(0).cuda()
view2['camera_intrinsics'] = view2['camera_intrinsics'].unsqueeze(0).cuda()

print("Running forward pass...")
with torch.no_grad():
    try:
        pred1, pred2 = model.forward(view1, view2)
        print("✓ Forward pass successful!")
        print(f"  pred1 keys: {pred1.keys()}")
        print(f"  pred1['pts3d'] shape: {pred1['pts3d'].shape}")
        print(f"  pred1['sh'] shape: {pred1['sh'].shape}")
        print(f"  pred1['scales'] shape: {pred1['scales'].shape}")
        print(f"  pred1['rotations'] shape: {pred1['rotations'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\nAll tests passed!")
