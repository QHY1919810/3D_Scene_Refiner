#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a 3DGS PLY from COLMAP TXT cameras (cameras_scene.txt / images_scene.txt).

This is the "render-only" half of colmap_to_scene_and_render_v7_gsrender_style.py:
  - It DOES NOT transform poses.
  - It ONLY reads the already-transformed COLMAP TXT model (scene coordinates)
    and renders each view using the gaussian-splatting renderer (gaussian_renderer.render).

Inputs
- combined_ply: merged scene(+car) 3DGS ply.
- cameras_txt: cameras_scene.txt (COLMAP TXT cameras)
- images_txt:  images_scene.txt  (COLMAP TXT images)

Outputs
- out_dir/*.png : rendered RGB images named by image basename (or by image id if duplicated).

Important notes
- Intrinsics are taken from cameras_txt. Distortion is NOT applied during rendering.
  If your COLMAP model uses a distorted camera model (OPENCV, RADIAL, etc.), you should
  use an undistorted COLMAP model.
- gaussian_renderer expects *transposed* w2c/projection matrices (gsrender style).
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np


# ----------------------------
# TXT parsing
# ----------------------------

def read_cameras_txt(path: str) -> Dict[int, Dict[str, Any]]:
    """
    COLMAP cameras.txt format (single-line per camera):
      CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    """
    cams: Dict[int, Dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        model = parts[1]
        w = int(parts[2])
        h = int(parts[3])
        params = [float(x) for x in parts[4:]]
        cams[cid] = {"id": cid, "model": model, "width": w, "height": h, "params": params}
    return cams


def read_images_txt(path: str) -> Dict[int, Dict[str, Any]]:
    """
    COLMAP images.txt format (two-line per image; second line is 2D points or blank):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      (blank or 2D points)
    """
    ims: Dict[int, Dict[str, Any]] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        iid = int(parts[0])
        q = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]  # w,x,y,z
        t = [float(parts[5]), float(parts[6]), float(parts[7])]
        cam_id = int(parts[8])
        name = " ".join(parts[9:])  # allow spaces (rare)
        ims[iid] = {"id": iid, "qvec": q, "tvec": t, "camera_id": cam_id, "name": name}
        # skip 2D line if present
        if i < len(lines) and lines[i].strip() == "":
            i += 1
    return ims


def qvec2rotmat_wxyz(q):
    """q = [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X; wY = w * Y; wZ = w * Z
    xX = x * X; xY = x * Y; xZ = x * Z
    yY = y * Y; yZ = y * Z; zZ = z * Z
    R = np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
    ], dtype=np.float64)
    return R


# ----------------------------
# Rendering (gaussian_renderer.render)
# ----------------------------

def fx_to_fov(fx: float, w: int) -> float:
    return float(2.0 * math.atan(float(w) / (2.0 * float(fx) + 1e-9)))


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    tan_half_fovx = math.tan(fovx * 0.5)
    tan_half_fovy = math.tan(fovy * 0.5)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / (tan_half_fovx + 1e-9)
    P[1, 1] = 1.0 / (tan_half_fovy + 1e-9)
    P[2, 2] = float(zfar) / (float(zfar) - float(znear) + 1e-9)
    P[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear) + 1e-9)
    P[3, 2] = 1.0
    return P


class SimpleCamera:
    """Minimal camera compatible with gaussian_renderer.render (gsrender style)."""
    def __init__(self, Rcw, tcw, FoVx, FoVy, H, W, device, znear=0.01, zfar=1000.0):
        import torch
        self.R = np.asarray(Rcw, dtype=np.float64)
        self.T = np.asarray(tcw, dtype=np.float64)
        self.FoVx = float(FoVx)
        self.FoVy = float(FoVy)
        self.image_height = int(H)
        self.image_width = int(W)
        self.data_device = device

        self.original_image = torch.zeros((3, self.image_height, self.image_width),
                                          dtype=torch.float32, device=device)
        self.gt_alpha_mask = None
        self.image_name = ""
        self.colmap_id = 0
        self.uid = 0

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.asarray(self.R, dtype=np.float32)
        w2c[:3, 3] = np.asarray(self.T, dtype=np.float32)

        Rm = np.asarray(self.R, dtype=np.float32)
        Tv = np.asarray(self.T, dtype=np.float32).reshape(3, 1)
        C = (-Rm.T @ Tv).reshape(3)

        P = get_projection_matrix(znear, zfar, self.FoVx, self.FoVy).astype(np.float32)

        # gaussian_renderer expects transposed matrices
        self.world_view_transform = torch.from_numpy(w2c).to(device=device, dtype=torch.float32).transpose(0, 1)
        self.projection_matrix = torch.from_numpy(P).to(device=device, dtype=torch.float32).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.from_numpy(C).to(device=device, dtype=torch.float32)


def render_from_txt(
    combined_ply: str,
    cameras_txt: str,
    images_txt: str,
    out_dir: str,
    device: str = "cuda",
    max_images: int = 0,
    bg_color=(0.0, 0.0, 0.0),
    znear: float = 0.01,
    zfar: float = 1000.0,
):
    import torch
    from PIL import Image

    try:
        from argparse import ArgumentParser
        from arguments import PipelineParams
        from utils.general_utils import safe_state
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
    except Exception as e:
        raise RuntimeError(
            "Cannot import gaussian-splatting modules. Run this inside your gaussian-splatting repo environment "
            "(so that scene/, gaussian_renderer/, arguments.py, utils/ are on PYTHONPATH). " + str(e)
        )

    cams = read_cameras_txt(cameras_txt)
    ims = read_images_txt(images_txt)
    if len(cams) == 0 or len(ims) == 0:
        raise ValueError("Parsed empty cameras/images. Check your *_scene.txt paths.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # PipelineParams provides pipe.antialiasing etc.
    safe_state(False)
    gs_parser = ArgumentParser(add_help=False)
    pp_group = PipelineParams(gs_parser)
    gs_args = gs_parser.parse_args([])
    pipe = pp_group.extract(gs_args)

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(combined_ply)
    if hasattr(gaussians, "active_sh_degree"):
        gaussians.active_sh_degree = getattr(gaussians, "max_sh_degree", 3)

    bg = torch.tensor(bg_color, dtype=torch.float32, device=dev)

    # render order: by image id
    image_ids = sorted(ims.keys())
    if max_images and max_images > 0:
        image_ids = image_ids[:max_images]

    used_names = set()
    for idx, iid in enumerate(image_ids):
        im = ims[iid]
        cam = cams[int(im["camera_id"])]

        W = int(cam["width"])
        H = int(cam["height"])
        model = cam["model"].upper()
        params = cam["params"]

        # Intrinsics -> FoV
        # PINHOLE: fx fy cx cy
        # SIMPLE_PINHOLE: fx cx cy
        # SIMPLE_RADIAL/RADIAL/OPENCV etc.: we ignore distortion; use fx/fy from params if available.
        if model in ("PINHOLE",):
            fx, fy = float(params[0]), float(params[1])
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = float(params[0]); fy = fx
        elif model in ("OPENCV", "OPENCV_FISHEYE"):
            fx, fy = float(params[0]), float(params[1])
        else:
            # fallback: treat first param as fx
            fx = float(params[0]); fy = fx

        fovx = fx_to_fov(fx, W)
        fovy = fx_to_fov(fy, H)

        Rcw = qvec2rotmat_wxyz(im["qvec"])
        tcw = np.asarray(im["tvec"], dtype=np.float64)

        viewpoint = SimpleCamera(Rcw, tcw, fovx, fovy, H, W, dev, znear=znear, zfar=zfar)
        viewpoint.colmap_id = int(iid)
        viewpoint.image_name = im["name"]

        render_pkg = render(viewpoint, gaussians, pipe, bg)
        rgb = render_pkg["render"]  # [3,H,W], float32 0..1
        rgb_np = (rgb.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # output name
        base = Path(im["name"]).stem
        out_name = f"{base}.png"
        if out_name in used_names:
            out_name = f"{base}_{iid}.png"
        used_names.add(out_name)

        Image.fromarray(rgb_np).save(str(Path(out_dir) / out_name))

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(image_ids)}] rendered {out_name}")

    print("DONE")
    print(f"Wrote renders to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_ply", type=str, required=True, help="Merged scene(+car) 3DGS ply to render.")
    ap.add_argument("--cameras_txt", type=str, required=True, help="cameras_scene.txt (COLMAP TXT).")
    ap.add_argument("--images_txt", type=str, required=True, help="images_scene.txt (COLMAP TXT).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for PNG renders.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_images", type=int, default=0, help="0=all, else render first N images (sorted by image id).")
    ap.add_argument("--bg", type=str, default="0,0,0", help="Background color r,g,b in 0..1.")
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=1000.0)
    args = ap.parse_args()

    bg = tuple(float(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg must be r,g,b")

    render_from_txt(
        combined_ply=args.combined_ply,
        cameras_txt=args.cameras_txt,
        images_txt=args.images_txt,
        out_dir=args.out_dir,
        device=args.device,
        max_images=args.max_images,
        bg_color=bg,
        znear=args.znear,
        zfar=args.zfar,
    )


if __name__ == "__main__":
    main()
