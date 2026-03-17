#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract "car-only" gaussians from a 3DGS point_cloud.ply using multi-view 2D masks.
[Enhanced Version with Strict Masking and 3D Open3D Post-processing]
"""

import argparse
import os
import struct
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# 新增依赖
import cv2


# -----------------------------
# COLMAP binary readers (Keep exactly as original)
# -----------------------------
@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # float64

@dataclass
class ImageEntry:
    id: int
    qvec: np.ndarray  # (4,)
    tvec: np.ndarray  # (3,)
    camera_id: int
    name: str

def read_cameras_text(path: str) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)
    return R

def read_images_text(path: str) -> Dict[int, ImageEntry]:
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            img_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
            images[img_id] = ImageEntry(img_id, qvec, tvec, camera_id, name)
            _ = f.readline()
    return images

def load_colmap_model(sparse_dir: str) -> Tuple[Dict[int, Camera], Dict[int, ImageEntry]]:
    cam_txt = os.path.join(sparse_dir, "cameras.txt")
    img_txt = os.path.join(sparse_dir, "images.txt")
    if not (os.path.exists(cam_txt) and os.path.exists(img_txt)):
        raise FileNotFoundError("Missing COLMAP TXT model.")
    return read_cameras_text(cam_txt), read_images_text(img_txt)

# -----------------------------
# Camera projection
# -----------------------------
def project_points(camera: Camera, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = (R @ Xw.T).T + t.reshape(1, 3)
    z = Xc[:, 2].copy()
    valid = z > 1e-6
    x = Xc[valid, 0] / z[valid]
    y = Xc[valid, 1] / z[valid]

    model = camera.model
    p = camera.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        u = f * x + cx; v = f * y + cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
        u = fx * x + cx; v = fy * y + cy
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        r2 = x*x + y*y
        radial = 1 + k * r2
        u = f * (x * radial) + cx; v = f * (y * radial) + cy
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p
        r2 = x*x + y*y
        radial = 1 + k1 * r2 + k2 * r2*r2
        u = f * (x * radial) + cx; v = f * (y * radial) + cy
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2
        x2 = x*x; y2 = y*y; xy = x*y
        x_dist = x*radial + 2*p1*xy + p2*(r2 + 2*x2)
        y_dist = y*radial + p1*(r2 + 2*y2) + 2*p2*xy
        u = fx * x_dist + cx; v = fy * y_dist + cy
    else:
        # Fallback for brevity
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        u = fx * x + cx; v = fy * y + cy

    uv = np.zeros((Xw.shape[0], 2), dtype=np.float64)
    uv[:] = np.nan
    uv[valid, 0] = u
    uv[valid, 1] = v
    return uv, z

# -----------------------------
# Enhanced SAM integration
# -----------------------------
def try_load_sam(sam_ckpt: Optional[str], sam_model_type: str):
    if sam_ckpt is None: return None
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
    except Exception as e:
        print("[WARN] segment-anything not available:", repr(e))
        return None
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"[OK] Loaded SAM ({sam_model_type}) on {device}")
    return predictor

def sam_mask_from_bbox_enhanced(predictor, image_rgb: np.ndarray, bbox_xyxy: np.ndarray, use_center: bool, erode_px: int) -> np.ndarray:
    """Enhanced SAM masking using Center Point + BBox + Erosion"""
    predictor.set_image(image_rgb)
    box = bbox_xyxy.astype(np.float32)
    
    if use_center:
        # 使用中心点作为强正向提示，避免 SAM 乱吃到边缘的阴影或马路
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        pts = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            box=box[None, :],
            multimask_output=True
        )
    else:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
        
    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8)

    # 腐蚀掉边缘的马路残余和阴影
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
    return mask

# -----------------------------
# 3D Post-Processing (Open3D)
# -----------------------------
def post_process_3d(xyz: np.ndarray, keep_idx: np.ndarray, args) -> np.ndarray:
    """Use Open3D to clean up road residues and floating artifacts."""
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] Open3D not installed. Skipping 3D post-processing.")
        return keep_idx

    print("\n[6] 3D Post-processing (Open3D) to remove road residue...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[keep_idx])

    # 1. Statistical Outlier Removal (去除零星飞点)
    if args.use_sor:
        print(f"    -> Running SOR (nb={args.sor_nb}, std={args.sor_std})...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=args.sor_nb, std_ratio=args.sor_std)
        keep_idx = keep_idx[ind]

    # 2. RANSAC Ground Removal (强力剔除车底的平坦路面)
    if args.remove_ground:
        print(f"    -> Running RANSAC Plane Removal (thr={args.ground_thr})...")
        plane_model, inliers = pcd.segment_plane(distance_threshold=args.ground_thr,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        if len(inliers) > 0:
            print(f"       Found road plane with {len(inliers)} points. Slicing them off.")
            ind = np.setdiff1d(np.arange(len(pcd.points)), inliers)
            pcd = pcd.select_by_index(ind)
            keep_idx = keep_idx[ind]

    # 3. DBSCAN (保留最大连通域，即车体，丢弃孤立的道路碎片)
    if args.use_dbscan:
        print(f"    -> Running DBSCAN (eps={args.dbscan_eps})...")
        labels = np.array(pcd.cluster_dbscan(eps=args.dbscan_eps, min_points=10, print_progress=False))
        if len(labels) > 0:
            largest_cluster_id = np.argmax(np.bincount(labels[labels >= 0]))
            ind = np.where(labels == largest_cluster_id)[0]
            pcd = pcd.select_by_index(ind)
            keep_idx = keep_idx[ind]

    print(f"    -> Points after 3D cleaning: {len(keep_idx):,}")
    return keep_idx

# -----------------------------
# Main Helpers
# -----------------------------
def load_ply_vertices(ply_path: str):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    return ply, v, xyz

def bbox_from_uv(uv: np.ndarray, W: int, H: int, pad=10) -> Optional[np.ndarray]:
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    if finite.sum() < 100: return None
    u = uv[finite, 0]; v = uv[finite, 1]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inb]; v = v[inb]
    if u.size < 200: return None
    x0 = np.quantile(u, 0.02) - pad; x1 = np.quantile(u, 0.98) + pad
    y0 = np.quantile(v, 0.02) - pad; y1 = np.quantile(v, 0.98) + pad
    return np.array([np.clip(x0, 0, W-1), np.clip(y0, 0, H-1), np.clip(x1, 0, W-1), np.clip(y1, 0, H-1)])

def make_bbox_mask(H: int, W: int, bbox: np.ndarray) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[int(bbox[1]):int(bbox[3])+1, int(bbox[0]):int(bbox[2])+1] = 1
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out_ply", default="car_only.ply")
    ap.add_argument("--n_views", type=int, default=30)
    ap.add_argument("--score_thr", type=float, default=0.7)
    ap.add_argument("--min_vis", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=200000)
    
    # --- 新增的强力清理参数 ---
    ap.add_argument("--mask_erode_px", type=int, default=5, help="Erode 2D mask by X pixels to cut off road edges.")
    ap.add_argument("--sam_use_center", action="store_true", help="Use bbox center as a positive prompt for SAM.")
    
    ap.add_argument("--use_sor", action="store_true", help="Enable Statistical Outlier Removal in 3D.")
    ap.add_argument("--sor_nb", type=int, default=20)
    ap.add_argument("--sor_std", type=float, default=2.0)
    
    ap.add_argument("--remove_ground", action="store_true", help="Enable RANSAC to detect and remove the road plane.")
    ap.add_argument("--ground_thr", type=float, default=0.03, help="RANSAC distance threshold (meters/units).")
    
    ap.add_argument("--use_dbscan", action="store_true", help="Enable DBSCAN to keep only the main car body.")
    ap.add_argument("--dbscan_eps", type=float, default=0.1, help="DBSCAN distance threshold.")
    # -------------------------

    ap.add_argument("--sam_ckpt", default=None)
    ap.add_argument("--sam_model", default="vit_h")
    ap.add_argument("--mask_dir", default="car_masks")
    ap.add_argument("--save_masks", action="store_true")
    args = ap.parse_args()

    print("[1] Load 3DGS ply...")
    ply, vtx, xyz = load_ply_vertices(args.ply)
    N = xyz.shape[0]

    print("[2] Load COLMAP TXT model...")
    cameras, images = load_colmap_model(args.sparse)
    image_items = sorted(images.values(), key=lambda e: e.name)
    sel_ids = np.linspace(0, len(image_items) - 1, args.n_views).round().astype(int)
    sel_entries = [image_items[i] for i in sel_ids]

    predictor = try_load_sam(args.sam_ckpt, args.sam_model)

    masks, proj_params = [], []
    print("[3] Generate enhanced per-view car masks...")
    for ent in tqdm(sel_entries):
        cam = cameras[ent.camera_id]
        with Image.open(os.path.join(args.images, ent.name)) as im:
            W, H = im.size
            image_rgb = np.array(im.convert("RGB"))

        R, t = qvec2rotmat(ent.qvec), ent.tvec
        Xsub = xyz[np.random.choice(N, size=min(N, 300000), replace=False)]
        uv_sub, _ = project_points(cam, R, t, Xsub)
        
        # 显式判断是否为 None，避免 NumPy 布尔歧义
        bbox = bbox_from_uv(uv_sub, W, H, pad=10)
        if bbox is None:
            bbox = np.array([0, 0, W-1, H-1], dtype=np.float64)


        if predictor is not None:
            mask = sam_mask_from_bbox_enhanced(predictor, image_rgb, bbox, args.sam_use_center, args.mask_erode_px)
        else:
            mask = make_bbox_mask(H, W, bbox)

        masks.append(mask)
        proj_params.append((cam, R, t, W, H, ent.name))

    print("[4] Vote gaussians by multi-view masks...")
    hits = np.zeros((N,), dtype=np.uint16)
    vis = np.zeros((N,), dtype=np.uint16)

    for start in tqdm(range(0, N, args.chunk)):
        end = min(N, start + args.chunk)
        X = xyz[start:end]
        lh, lv = np.zeros(end-start, dtype=np.uint16), np.zeros(end-start, dtype=np.uint16)

        for (cam, R, t, W, H, name), mask in zip(proj_params, masks):
            uv, z = project_points(cam, R, t, X)
            u, v = uv[:, 0], uv[:, 1]
            finite = np.isfinite(u) & np.isfinite(v) & (z > 1e-6)
            if not np.any(finite): continue
            
            ui, vi = u[finite].astype(np.int32), v[finite].astype(np.int32)
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not np.any(inb): continue

            idx_chunk = np.nonzero(finite)[0][inb]
            ui, vi = ui[inb], vi[inb]

            lv[idx_chunk] += 1
            lh[idx_chunk] += mask[vi, ui]

        hits[start:end] = lh
        vis[start:end] = lv

    score = hits.astype(np.float64) / np.maximum(vis.astype(np.float64), 1.0)
    keep = (vis >= args.min_vis) & (score >= args.score_thr)
    keep_idx = np.nonzero(keep)[0]
    print(f"    Initial Keep: {keep_idx.size:,} / {N:,}")

    # --- 强力清洗 3D 拓扑结构 ---
    keep_idx = post_process_3d(xyz, keep_idx, args)

    print("\n[7] Write filtered ply...")
    v_keep = vtx[keep_idx]
    new_el = PlyElement.describe(v_keep, "vertex")
    PlyData([new_el], text=ply.text).write(args.out_ply)
    print(f"[OK] Done. Output: {args.out_ply}")

if __name__ == "__main__":
    main()