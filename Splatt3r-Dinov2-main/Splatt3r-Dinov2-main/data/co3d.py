import os
import json
import gzip
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CO3DDataset(Dataset):
    def __init__(self, root_dir, category, split='train', img_size=(512, 512), frame_gap=(5, 10), **kwargs):
        """
        Args:
            root_dir: Path to the downloaded CO3D data (e.g., ../co3dv2_single)
            category: IS IGNORED (set to null in config)
            split: 'train' or 'test'
            img_size: Tuple (H, W)
            frame_gap: Tuple (min_gap, max_gap) for sampling pairs
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.min_gap, self.max_gap = frame_gap

        self.sequences = {}
        self.pairs = []

        # 1. Find all 'frame_annotations.jgz' files
        print(f"[CO3D] Scanning for .jgz files in: {root_dir}")
        ann_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file == 'frame_annotations.jgz':
                    ann_files.append(os.path.join(root, file))
        
        if not ann_files:
            raise FileNotFoundError(f"Could not find any 'frame_annotations.jgz' in {root_dir}. Did the download finish unpacking?")
        
        print(f"[CO3D] Loading {len(ann_files)} annotation files...")

        # 2. Load all frames from all found annotation files
        all_frames = []
        for ann_path in ann_files:
            try:
                with gzip.open(ann_path, 'rt', encoding='utf-8') as f:
                    frames = json.load(f)
                    
                    # === FIX IS HERE ===
                    # The frame['image']['path'] in the JSON *already* contains the category
                    # (e.g., "bottle/images/frame.jpg"). We don't need to add it.
                    all_frames.extend(frames) 
                    # ===================

            except Exception as e:
                print(f"Warning: Failed to read {ann_path}. Error: {e}")
        
        if not all_frames:
             raise RuntimeError(f"Failed to load any frames from {len(ann_files)} annotation files.")

        # 3. Group frames by Sequence
        for frame in all_frames:
            seq_name = frame['sequence_name']
            if seq_name not in self.sequences:
                self.sequences[seq_name] = []
            self.sequences[seq_name].append(frame)

        # 4. Split into Train/Test
        seq_names = sorted(list(self.sequences.keys()))
        split_idx = int(0.9 * len(seq_names))
        
        if split == 'train':
            self.selected_seqs = seq_names[:split_idx]
        else: # 'test' or 'val'
            self.selected_seqs = seq_names[split_idx:]
        
        print(f"[CO3D] Found {len(seq_names)} total sequences. Using {len(self.selected_seqs)} for {split}.")

        # 5. Pre-compute Pairs
        for seq in self.selected_seqs:
            frames = self.sequences[seq]
            frames.sort(key=lambda x: x['frame_number'])
            
            valid_frames = []
            for f in frames:
                # f['image']['path'] is now "bottle/images/frame.jpg"
                full_path = os.path.join(self.root_dir, f['image']['path'])
                if os.path.exists(full_path):
                    valid_frames.append(f)
            
            if len(valid_frames) < 2:
                continue

            for i in range(len(valid_frames)):
                for gap in range(self.min_gap, self.max_gap + 1):
                    if i + gap < len(valid_frames):
                        self.pairs.append((valid_frames[i], valid_frames[i+gap]))

        print(f"[CO3D] Generated {len(self.pairs)} pairs for {split}.")
        if len(self.pairs) == 0:
            print("WARNING: 0 pairs generated. Check dataset path and frame gaps.")

        # 6. Normalization
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame1, frame2 = self.pairs[idx]
        
        path1 = os.path.join(self.root_dir, frame1['image']['path'])
        path2 = os.path.join(self.root_dir, frame2['image']['path'])

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        K1 = self._get_intrinsics(frame1)
        K2 = self._get_intrinsics(frame2)
        
        pose1 = self._get_pose(frame1)
        pose2 = self._get_pose(frame2)

        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        # Extract sequence name and frame numbers
        sequence_name = frame1.get('sequence_name', 'default')
        frame_num1 = str(frame1.get('frame_number', 0))
        frame_num2 = str(frame2.get('frame_number', 0))
        
        # Create view dicts matching ScanNet++ format
        view1 = {
            'img': img1_tensor, 
            'original_img': img1_tensor,  # ScanNet++ has this
            'camera_pose': pose1, 
            'camera_intrinsics': K1, 
            'instance': frame_num1,  # Instance should be frame number for is_symmetrized to work
        }
        
        view2 = {
            'img': img2_tensor, 
            'original_img': img2_tensor,  # ScanNet++ has this
            'camera_pose': pose2, 
            'camera_intrinsics': K2, 
            'instance': frame_num2,  # Instance should be frame number for is_symmetrized to work
        }
        
        # Return in ScanNet++ format: context views + target views
        return {
            'context': [view1, view2],  # Changed from 'view1'/'view2' to 'context' list
            'target': [view2],  # Empty target for now (used during training only)
            'scene': sequence_name
        }

    def _get_pose(self, frame_meta):
        vp = frame_meta['viewpoint']
        R = torch.tensor(vp['R'])
        T = torch.tensor(vp['T'])
        pose = torch.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        return pose

    def _get_intrinsics(self, frame_meta):
        vp = frame_meta['viewpoint']
        focal = vp['focal_length']
        principal = vp['principal_point']
        
        H_orig, W_orig = frame_meta['image']['size']
        if H_orig == 0 or W_orig == 0: # Safety check for bad metadata
             H_orig, W_orig = self.img_size[0], self.img_size[1]
             
        # Guard against bad metadata (e.g., divide by zero)
        if H_orig == 0: H_orig = self.img_size[0]
        if W_orig == 0: W_orig = self.img_size[1]
             
        scale_h = self.img_size[0] / H_orig
        scale_w = self.img_size[1] / W_orig
        
        K = torch.eye(3)
        K[0, 0] = focal[0] * scale_w
        K[1, 1] = focal[1] * scale_h
        K[0, 2] = principal[0] * scale_w
        K[1, 2] = principal[1] * scale_h
        
        return K