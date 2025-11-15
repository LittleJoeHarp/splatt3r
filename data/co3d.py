import gzip
import io
from pathlib import Path
import numpy as np
import PIL.Image
from .data import DUST3RSplattingDataset, DUST3RSplattingTestDataset

class Co3dData:
    def __init__(self, root, split_path):
        self.root = Path(root)

        # Load all sequences from train.txt/val.txt
        with open(split_path, 'r') as f:
            all_sequences = [line.strip() for line in f.readlines()]

        # Initialize storage
        self.sequences = []
        self.color_paths = {}    # list of image path strings (may include .gz)
        self.poses = {}
        self.intrinsics = {}

        # Supported image extensions (uncompressed and compressed)
        img_exts = [".png", ".jpg", ".jpeg", ".png.gz", ".jpg.gz", ".jpeg.gz"]

        # --- Filter and load sequences ---
        for seq in all_sequences:
            seq_dir = self.root / seq

            # Find images (search rgb/ then images/ then root)
            possible_dirs = [seq_dir / "rgb", seq_dir / "images", seq_dir]
            imgs = []
            for d in possible_dirs:
                if d.exists() and d.is_dir():
                    for ext in img_exts:
                        imgs.extend(sorted([p for p in d.glob(f"*{ext}")]))
                    if imgs:
                        break

            # Require at least 2 images to sample pairs
            if len(imgs) < 2:
                continue

            # 2. Poses/Intrinsics Load (supports .gz)
            try:
                # Poses
                pose_file_path = seq_dir / "poses.txt"
                if not pose_file_path.exists():
                    pose_file_path = seq_dir / "poses.txt.gz"
                if not pose_file_path.exists():
                    continue

                if str(pose_file_path).endswith('.gz'):
                    with gzip.open(pose_file_path, 'rt') as f:
                        poses_raw = np.loadtxt(f, dtype=np.float64)
                else:
                    poses_raw = np.loadtxt(pose_file_path, dtype=np.float64)

                poses = poses_raw.reshape(-1, 4, 4).astype(np.float32)

                # Intrinsics
                intrin_file_path = seq_dir / "intrinsics.txt"
                if not intrin_file_path.exists():
                    intrin_file_path = seq_dir / "intrinsics.txt.gz"
                if not intrin_file_path.exists():
                    continue

                if str(intrin_file_path).endswith('.gz'):
                    with gzip.open(intrin_file_path, 'rt') as f:
                        fx, fy, cx, cy = np.loadtxt(f, dtype=np.float64)
                else:
                    fx, fy, cx, cy = np.loadtxt(intrin_file_path, dtype=np.float64)

            except Exception:
                # Skip corrupted/unreadable sequences
                continue

            # Everything ok -> add sequence
            self.sequences.append(seq)
            # store paths as strings (Path objects also fine) in sorted order
            self.color_paths[seq] = [str(p) for p in imgs]
            self.poses[seq] = poses

            # Prepare Intrinsics (3x3)
            K = np.eye(3, dtype=np.float32)
            K[0,0], K[1,1] = np.float32(fx), np.float32(fy)
            K[0,2], K[1,2] = np.float32(cx), np.float32(cy)
            self.intrinsics[seq] = K

    def _open_image_path(self, img_path):
        """
        Open image at img_path which can be:
          - uncompressed .png/.jpg/.jpeg (PIL opens directly)
          - compressed .png.gz/.jpg.gz/.jpeg.gz (we read bytes via gzip then PIL)
        Returns a PIL.Image (RGB).
        """
        p = str(img_path)
        if p.endswith(".gz"):
            # read compressed bytes and open via BytesIO
            with gzip.open(p, "rb") as fh:
                data = fh.read()
            return PIL.Image.open(io.BytesIO(data)).convert("RGB")
        else:
            return PIL.Image.open(p).convert("RGB")

    def get_view(self, sequence, view_idx, resolution):
        img_path = self.color_paths[sequence][view_idx]
        # Open image (handles gz on-the-fly)
        original_img = self._open_image_path(img_path)
        # Resize to resolution (PIL size is (W,H))
        if original_img.size != tuple(resolution):
            original_img = original_img.resize(tuple(resolution), PIL.Image.LANCZOS)

        # Camera pose: input poses are world->camera; invert to camera->world
        camera_pose = self.poses[sequence][view_idx].copy()
        camera_pose = np.linalg.inv(camera_pose).astype(np.float32)

        # Intrinsics (already float32)
        intrinsics = self.intrinsics[sequence]

        # Depth placeholder (H,W)
        H, W = resolution[1], resolution[0]
        depthmap = np.zeros((H, W), dtype=np.float32)

        # Instance id
        instance_id = f"{sequence}_{view_idx}"

        return {
            "original_img": original_img,
            "depthmap": depthmap,
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "true_shape": np.array(original_img.size),
            "instance": instance_id,
            "is_metric_scale": False,
            "scale": 1.0,
            "dataset": "co3d",
            "sky_mask": np.zeros((H, W), dtype=bool),
        }

def get_co3d_dataset(root, split, resolution, num_epochs_per_epoch=1):
    data_source = Co3dData(root, split)

    # Check if any sequences survived the filtering
    if not data_source.sequences:
        raise ValueError(f"No valid sequences found in {split} list after filtering (need >=2 images).")

    coverage = {}
    for seq in data_source.sequences:
        if seq not in data_source.color_paths: continue
        n = len(data_source.color_paths[seq])
        mat = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(max(0,i-5), min(n,i+5)):
                mat[i,j] = 0.8
        coverage[seq] = mat

    return DUST3RSplattingDataset(data_source, coverage, resolution, num_epochs_per_epoch=num_epochs_per_epoch)

def get_co3d_test_dataset(root, split, resolution):
    data_source = Co3dData(root, split)

    # Check if any sequences survived the filtering
    if not data_source.sequences:
        # Return an empty dataset wrapper if nothing survived, to prevent crash
        print(f"Warning: No valid sequences found in {split} list for testing.")
        return DUST3RSplattingTestDataset(data_source, [], resolution)

    samples = []
    for seq in data_source.sequences:
        if seq not in data_source.color_paths: continue
        n = len(data_source.color_paths[seq])
        # We know n > 1 due to filtering in __init__
        samples.append((seq, 0, 2, 1)) # Context 0,2 -> Target 1
        if n > 12:
            samples.append((seq, 10, 12, 11))

    return DUST3RSplattingTestDataset(data_source, samples, resolution)
