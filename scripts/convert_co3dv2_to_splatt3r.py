#!/usr/bin/env python3
"""
Low-space CO3D -> Splatt3r converter.

Strategies to reduce disk usage (configurable via SAVE_MODE):
 - "jpeg"  : save resized JPEGs (quality, max dim configurable)
 - "hardlink": create hard links to original files when on same FS (falls back to copy)
 - "symlink": create symlinks (small, but depends on original files staying in place)
 - "none"  : do not save images; write image_paths.txt referencing original images

Also compresses poses/intrinsics with gzip (.txt.gz).
Two-pass per-sequence: validate frames first, only write when sequence passes MIN_FRAMES.
"""
import gzip
import json
import shutil
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image

# Paths (adjust)
CO3D_ROOT = Path("/home/ritama/datasets/co3dv2_single")
OUT_ROOT = Path("/home/ritama/datasets/co3d_processed")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Config: tune these to trade quality vs space
MIN_FILE_SIZE_KB = 1
MIN_FRAMES = 3
# image saving strategy: "jpeg" | "hardlink" | "symlink" | "none"
SAVE_MODE = "jpeg"
JPEG_QUALITY = 80        # 0-100 (lower => smaller files)
MAX_DIM = 512            # max width/height when resizing for JPEG (preserve aspect)
COMPRESS_META = True     # write poses.txt.gz and intrinsics.txt.gz if True

# Simple black detection thresholds (tweak as needed)
MIN_PIXEL_MEAN = 5.0
DARK_PIXEL_THRESHOLD = 16
DARK_PIXEL_FRACTION = 0.98
MIN_STDDEV = 2.0

def is_black_image(p: Path):
    try:
        im = Image.open(p).convert("L")
        a = np.asarray(im, dtype=np.uint8)
        mean = float(a.mean())
        std = float(a.std())
        dark_frac = float((a <= DARK_PIXEL_THRESHOLD).mean())
        if mean < MIN_PIXEL_MEAN or std < MIN_STDDEV or dark_frac >= DARK_PIXEL_FRACTION:
            return True
        return False
    except Exception:
        return False  # be conservative and treat unreadable as non-black

def matrix4_to_string(M):
    M = np.array(M)
    try:
        M = M.reshape((4, 4))
    except Exception:
        flat = M.reshape(-1)[:16]
        if flat.size < 16:
            flat = np.pad(flat, (0, 16 - flat.size), mode='constant', constant_values=0.0)
        M = flat.reshape((4, 4))
    return " ".join(map(str, M.reshape(-1).tolist()))

def resolve_path(cat_path: Path, rel_path):
    if rel_path is None:
        return None
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    rel = str(rel_path).lstrip("./")
    cand = cat_path / rel
    if cand.exists():
        return cand
    cand2 = CO3D_ROOT / rel
    if cand2.exists():
        return cand2
    # fallback: search by basename
    for f in cat_path.rglob(Path(rel).name):
        return f
    return cand

# Minimal intrinsics/extrinsics helpers (reuse logic from original)
def _get_intrinsics_from_dict(d):
    if d is None:
        return None
    fx_val = d.get("fx_px") or d.get("focal_length_px") or d.get("focal_length")
    fy_val = d.get("fy_px") or d.get("focal_length_px") or fx_val
    if isinstance(fx_val, (list, tuple)) and len(fx_val) >= 2:
        fx_val, fy_val = fx_val[0], fx_val[1]
    pp = d.get("principal_point_px") or d.get("principal_point")
    if isinstance(pp, (list, tuple)) and len(pp) >= 2:
        cx, cy = pp[0], pp[1]
    else:
        cx = d.get("cx") or d.get("principal_x") or None
        cy = d.get("cy") or d.get("principal_y") or None
    if fx_val is None or cx is None:
        return None
    return (float(fx_val), float(fy_val), float(cx), float(cy))

def _get_matrix_from_obj(obj):
    if obj is None:
        return None
    try:
        arr = np.array(obj, dtype=float)
        if arr.size == 16:
            return arr.reshape((4,4))
        if getattr(arr, "shape", None) == (4,4):
            return arr
        if getattr(arr, "shape", None) == (3,4):
            ext = np.eye(4, dtype=float)
            ext[:3,:4] = arr
            return ext
    except Exception:
        pass
    return None

def get_camera_data(frame):
    if "viewpoint" in frame:
        vp = frame["viewpoint"]
        intr = _get_intrinsics_from_dict(vp)
        if intr is None:
            val = vp.get("focal_length") or vp.get("focal_length_px")
            if isinstance(val, (list,tuple)):
                fx,fy = val[0], val[1]
            elif val is not None:
                fx = fy = val
            else:
                return None, None
            pp = vp.get("principal_point_px") or vp.get("principal_point")
            if isinstance(pp, (list,tuple)) and len(pp)>=2:
                cx,cy = pp[0], pp[1]
            else:
                return None, None
            intr = (float(fx), float(fy), float(cx), float(cy))
        R = np.array(vp.get("R")) if vp.get("R") is not None else None
        T = np.array(vp.get("T")) if vp.get("T") is not None else None
        if R is None or T is None:
            ext = _get_matrix_from_obj(vp.get("extrinsics") or vp.get("matrix"))
            return intr, ext
        ext = np.eye(4, dtype=float); ext[:3,:3]=np.array(R); ext[:3,3]=np.array(T)
        return intr, ext
    if "camera" in frame:
        cam = frame["camera"]
        intr = _get_intrinsics_from_dict(cam.get("intrinsics") or cam)
        ext_obj = cam.get("extrinsics") or frame.get("camera_extrinsics") or cam
        ext = _get_matrix_from_obj(ext_obj.get("matrix_world_to_camera") if isinstance(ext_obj, dict) else ext_obj)
        if intr is None:
            return None, None
        return intr, ext
    intr = _get_intrinsics_from_dict(frame.get("intrinsics") or frame.get("camera_intrinsics"))
    ext = _get_matrix_from_obj(frame.get("extrinsics") or frame.get("camera_extrinsics") or frame.get("matrix_world_to_camera"))
    if intr is not None:
        return intr, ext
    return None, None

def extract_image_path_from_frame(frame):
    im = frame.get("image")
    if isinstance(im, dict):
        return im.get("path") or im.get("file_name") or im.get("file_path")
    if isinstance(im, str):
        return im
    return frame.get("image_path") or frame.get("file_path") or frame.get("path")

def save_image_as_jpeg(src: Path, dst: Path, max_dim=MAX_DIM, quality=JPEG_QUALITY):
    try:
        im = Image.open(src).convert("RGB")
        w,h = im.size
        scale = min(1.0, float(max_dim) / max(w,h))
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=quality, optimize=True, progressive=True)
        return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False

def write_text(path: Path, text: str, compress: bool = True):
    if compress:
        with gzip.open(str(path)+".gz", "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text)

def process_category(cat_path: Path):
    frame_file = cat_path / "frame_annotations.jgz"
    if not frame_file.exists():
        print(f"no frame_annotations for {cat_path}")
        return
    with gzip.open(frame_file, "rb") as f:
        try:
            all_frames = json.load(f)
        except Exception as e:
            print(f"Failed to read {frame_file}: {e}")
            return
    frames_by_seq = defaultdict(list)
    for frame in all_frames:
        seq = frame.get("sequence_name") or frame.get("sequence") or frame.get("sequence_id") or "unknown"
        frames_by_seq[seq].append(frame)

    saved = 0
    for seq_name, frames in tqdm(frames_by_seq.items(), desc=f"Processing {cat_path.name}"):
        out_scene = OUT_ROOT / f"{cat_path.name}_{seq_name}"
        # First pass: collect candidates (no writes)
        candidates = []
        valid_intr = None
        for frame in frames:
            intr, ext = get_camera_data(frame)
            if ext is None:
                continue
            if valid_intr is None:
                valid_intr = intr
            img_rel = extract_image_path_from_frame(frame)
            img_src = resolve_path(cat_path, img_rel)
            if not img_src or not img_src.exists():
                continue
            if img_src.stat().st_size / 1024.0 < MIN_FILE_SIZE_KB:
                continue
            if is_black_image(img_src):
                continue
            candidates.append((img_src, ext))
        if len(candidates) < MIN_FRAMES or valid_intr is None:
            continue  # skip sequence, no writes performed

        # Sequence accepted: write outputs according to SAVE_MODE
        try:
            if out_scene.exists():
                shutil.rmtree(out_scene)
            out_scene.mkdir(parents=True, exist_ok=True)
            pose_lines = []
            image_paths_written = []
            if SAVE_MODE != "none":
                img_dir = out_scene / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
            for i, (src, ext) in enumerate(candidates):
                pose_lines.append(matrix4_to_string(ext))
                if SAVE_MODE == "jpeg":
                    dst = img_dir / f"{i:06d}.jpg"
                    ok = save_image_as_jpeg(src, dst)
                    if ok:
                        image_paths_written.append(str(dst.name))
                elif SAVE_MODE == "hardlink":
                    dst = img_dir / f"{i:06d}" + src.suffix
                    try:
                        os.link(src, dst)
                    except Exception:
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            continue
                    image_paths_written.append(str(dst.name))
                elif SAVE_MODE == "symlink":
                    dst = img_dir / f"{i:06d}" + src.suffix
                    try:
                        os.symlink(os.path.relpath(src, img_dir), dst)
                        image_paths_written.append(str(dst.name))
                    except Exception:
                        try:
                            shutil.copy2(src, dst)
                            image_paths_written.append(str(dst.name))
                        except Exception:
                            continue
                elif SAVE_MODE == "none":
                    # Just record original path
                    image_paths_written.append(str(src))
            # write poses and intrinsics (compressed if requested)
            write_text(out_scene / "poses.txt", "\n".join(pose_lines), compress=COMPRESS_META)
            fx,fy,cx,cy = valid_intr
            write_text(out_scene / "intrinsics.txt", f"{fx} {fy} {cx} {cy}\n", compress=COMPRESS_META)
            # write image_paths if not writing images
            if SAVE_MODE == "none":
                write_text(out_scene / "image_paths.txt", "\n".join(image_paths_written), compress=COMPRESS_META)
            saved += 1
        except Exception as e:
            print(f"Failed to write sequence {out_scene}: {e}")
            try:
                if out_scene.exists():
                    shutil.rmtree(out_scene)
            except Exception:
                pass
    print(f"Saved {saved} sequences for category {cat_path.name}")

def main():
    cats = sorted([p for p in CO3D_ROOT.iterdir() if p.is_dir()])
    for cat in cats:
        if cat.name in ['eval_batches', 'set_lists', '_in_progress']:
            continue
        process_category(cat)

if __name__ == "__main__":
    main()