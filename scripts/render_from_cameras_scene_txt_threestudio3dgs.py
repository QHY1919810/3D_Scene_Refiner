#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render from COLMAP TXT cameras (cameras_scene.txt / images_scene.txt) using *threestudio-3dgs* renderer.

This is the threestudio-3dgs equivalent of gaussian-splatting's gaussian_renderer.render.
It uses the fork's gsrender-style batch camera pipeline:
  - GaussianBatchRenderer.batch_forward expects batch["c2w"], batch["fovy"], batch["width"], batch["height"]
    and constructs Camera(FoVx/FoVy/world_view_transform/full_proj_transform) internally. (See gaussian_batch_renderer.py)
  - The Camera structure is defined in gaussian_base.py as a NamedTuple with FoVx/FoVy/camera_center/... (gsrender style)

What you provide:
  - combined_ply: merged scene(+car) 3DGS ply
  - cameras_txt: cameras_scene.txt
  - images_txt:  images_scene.txt

What it outputs:
  - out_dir/*.png rendered images

Assumptions / integration notes:
  1) Run this inside your threestudio repo environment (so `import threestudio` and `custom.threestudio-3dgs` works).
  2) This script instantiates:
       - GaussianBaseModel (geometry) and loads the .ply via `geometry_convert_from`.
       - The renderer registered as "diff-gaussian-rasterizer-shading".
     If your fork uses a different renderer key, override via --renderer_type.
  3) The renderer's forward() implementation may expect the geometry object under a specific kwarg name.
     We pass several common aliases in batch: gaussians/geometry/pc.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


# ----------------------------
# COLMAP TXT parsing
# ----------------------------

def read_cameras_txt(path: str) -> Dict[int, Dict[str, Any]]:
    cams: Dict[int, Dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        w = int(parts[2])
        h = int(parts[3])
        params = [float(x) for x in parts[4:]]
        cams[cid] = {"id": cid, "model": model, "width": w, "height": h, "params": params}
    return cams


def read_images_txt(path: str) -> Dict[int, Dict[str, Any]]:
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
        name = " ".join(parts[9:])
        ims[iid] = {"id": iid, "qvec": q, "tvec": t, "camera_id": cam_id, "name": name}
        # skip optional 2D line
        if i < len(lines) and lines[i].strip() == "":
            i += 1
    return ims


def qvec2rotmat_wxyz(q):
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


def fx_to_fov(fx: float, w: int) -> float:
    return float(2.0 * math.atan(float(w) / (2.0 * float(fx) + 1e-9)))


def w2c_to_c2w(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    """Rcw/tcw (world->cam) -> c2w 4x4 (cam->world)."""
    Rcw = np.asarray(Rcw, dtype=np.float64)
    tcw = np.asarray(tcw, dtype=np.float64).reshape(3)
    Rwc = Rcw.T
    C = (-Rwc @ tcw.reshape(3, 1)).reshape(3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rwc.astype(np.float32)
    c2w[:3, 3] = C.astype(np.float32)
    return c2w


# ----------------------------
# threestudio-3dgs instantiation helpers
# ----------------------------

def _ensure_repo_import(repo_root: str):
    if repo_root and os.path.isdir(repo_root):
        import sys
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)


def build_geometry(ply_path: str, sh_degree: int, device: torch.device):
    # GaussianBaseModel is registered as "gaussian-splatting" in gaussian_base.py
    import threestudio
    GeoCls = threestudio.find("gaussian-splatting")
    geo = GeoCls({
        "sh_degree": sh_degree,
        "geometry_convert_from": ply_path,
        "load_ply_only_vertex": False,
        # init fields unused since we load ply
        "init_num_pts": 100,
        "pc_init_radius": 0.8,
        "opacity_init": 0.1,
    })
    geo.to(device=device.type)
    return geo


def build_renderer(renderer_type: str, device: torch.device, bg_rgb=(0.0, 0.0, 0.0)):
    import threestudio
    RenCls = threestudio.find(renderer_type)
    renderer = RenCls({"debug": False})
    # GaussianBatchRenderer.forward expects self.background_tensor in this fork
    renderer.background_tensor = torch.tensor(bg_rgb, dtype=torch.float32, device=device)
    return renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/nfs4/qhy/projects/threestudio",
                    help="threestudio repo root (so custom/threestudio-3dgs is importable)")
    ap.add_argument("--combined_ply", type=str, required=True, help="Merged scene(+car) 3DGS ply.")
    ap.add_argument("--cameras_txt", type=str, required=True, help="cameras_scene.txt")
    ap.add_argument("--images_txt", type=str, required=True, help="images_scene.txt")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for PNG renders")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_images", type=int, default=0, help="0=all; else render first N image ids.")
    ap.add_argument("--bg", type=str, default="0,0,0", help="Background RGB in 0..1: r,g,b")
    ap.add_argument("--renderer_type", type=str, default="diff-gaussian-rasterizer-shading",
                    help='threestudio registry key for renderer (default matches your YAML).')
    ap.add_argument("--sh_degree", type=int, default=3, help="SH degree for loaded gaussians")
    args = ap.parse_args()

    _ensure_repo_import(args.repo_root)

    bg = tuple(float(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg must be r,g,b in 0..1")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cams = read_cameras_txt(args.cameras_txt)
    ims = read_images_txt(args.images_txt)
    if len(cams) == 0 or len(ims) == 0:
        raise ValueError("Parsed empty cameras/images. Check your *_scene.txt paths.")

    # Build geometry + renderer
    geometry = build_geometry(args.combined_ply, sh_degree=args.sh_degree, device=device)
    renderer = build_renderer(args.renderer_type, device=device, bg_rgb=bg)

    # Attach geometry in common attribute names (different forks use different naming)
    for attr in ["gaussians", "geometry", "pc"]:
        if hasattr(renderer, attr):
            setattr(renderer, attr, geometry)

    from PIL import Image

    image_ids = sorted(ims.keys())
    if args.max_images and args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    used = set()
    for idx, iid in enumerate(image_ids):
        im = ims[iid]
        cam = cams[int(im["camera_id"])]
        W = int(cam["width"])
        H = int(cam["height"])
        model = cam["model"].upper()
        params = cam["params"]

        # intrinsics -> FoV (distortion ignored)
        if model in ("PINHOLE",):
            fx, fy = float(params[0]), float(params[1])
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = float(params[0]); fy = fx
        elif model in ("OPENCV", "OPENCV_FISHEYE"):
            fx, fy = float(params[0]), float(params[1])
        else:
            fx = float(params[0]); fy = fx

        fovy = fx_to_fov(fy, H)  # radians

        Rcw = qvec2rotmat_wxyz(im["qvec"])
        tcw = np.asarray(im["tvec"], dtype=np.float64)
        c2w = w2c_to_c2w(Rcw, tcw)

        batch = {
            "c2w": torch.from_numpy(c2w).unsqueeze(0).to(device=device, dtype=torch.float32),
            "fovy": torch.tensor([fovy], device=device, dtype=torch.float32),
            "width": W,
            "height": H,
            # common aliases the renderer might look for
            "gaussians": geometry,
            "geometry": geometry,
            "pc": geometry,
        }

        # batch_forward produces comp_rgb [B,H,W,3]
        with torch.no_grad():
            out = renderer.batch_forward(batch)
            rgb = out["comp_rgb"][0].detach().clamp(0, 1).cpu().numpy()  # HWC float

        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        base = Path(im["name"]).stem
        out_name = f"{base}.png"
        if out_name in used:
            out_name = f"{base}_{iid}.png"
        used.add(out_name)

        Image.fromarray(rgb_u8).save(str(Path(args.out_dir) / out_name))

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(image_ids)}] rendered {out_name}")

    print("DONE")
    print(f"Wrote renders to: {args.out_dir}")


if __name__ == "__main__":
    main()
