#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP -> Scene camera transform (TXT-only)

This is a stripped-down version of `colmap_to_scene_and_render_v7_gsrender_style.py`:
- Reads COLMAP sparse model (cameras.bin, images.bin)
- Rebuilds the SAME Sim(3) insertion transform used to place the car into the scene:
    X_scene = s * R_total * X_car + t0
- Transforms each COLMAP camera pose into the scene coordinate system
- Writes ONLY:
    - cameras_scene.txt
    - images_scene.txt
    - transform_summary.json (optional but useful)

NO rendering is performed.

Notes:
- Intrinsics are copied from cameras.bin. Distortion is NOT applied anywhere.
  If your COLMAP model uses distortion (OPENCV/RADIAL/etc.), you should use an undistorted model.
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from plyfile import PlyData


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_ply_xyz(ply_path: str) -> np.ndarray:
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def dense_center_radius_voxel(
    xyz: np.ndarray,
    voxel_size: float = 1.0,
    topk_voxels: int = 200,
    clip_q: float = 0.02,
    radius_factor: float = 1.0,
):
    """Robust pivot from dense voxels (mirrors v7 logic enough for pivot reconstruction)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return np.zeros(3), np.zeros(3), 1.0, 0

    q = float(np.clip(clip_q, 0.0, 0.49))
    lo = np.quantile(xyz, q, axis=0)
    hi = np.quantile(xyz, 1.0 - q, axis=0)
    xyz_clip = np.clip(xyz, lo, hi)

    vox = np.floor(xyz_clip / float(voxel_size)).astype(np.int64)
    keys = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
    uniq, cnt = np.unique(keys, return_counts=True)
    k = int(min(max(topk_voxels, 1), uniq.shape[0]))
    top_idx = np.argsort(-cnt)[:k]
    top_keys = set(uniq[top_idx].tolist())
    sel = np.array([kk in top_keys for kk in keys], dtype=bool)
    xyz_sel = xyz_clip[sel]
    if xyz_sel.shape[0] == 0:
        xyz_sel = xyz_clip

    center = np.mean(xyz_sel, axis=0)
    r = float(np.max(np.linalg.norm(xyz_sel - center[None, :], axis=1)))
    r = max(r, 1e-6) * float(radius_factor)
    ext = hi - lo
    return center, ext, r, int(xyz_sel.shape[0])


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat2qvec_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


def _read_next_bytes(fid, num_bytes, fmt, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + fmt, data)


# COLMAP camera model mappings (common ones)
COLMAP_CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}
COLMAP_CAMERA_MODEL_NUM_PARAMS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}


def read_cameras_bin(path: str) -> Dict[int, Dict[str, Any]]:
    cameras: Dict[int, Dict[str, Any]] = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id = _read_next_bytes(fid, 4, "I")[0]
            model_id = _read_next_bytes(fid, 4, "i")[0]
            width = _read_next_bytes(fid, 8, "Q")[0]
            height = _read_next_bytes(fid, 8, "Q")[0]
            model_name = COLMAP_CAMERA_MODEL_IDS.get(model_id, "UNKNOWN")
            num_params = COLMAP_CAMERA_MODEL_NUM_PARAMS.get(model_id)
            if num_params is None:
                raise ValueError(f"Unknown COLMAP camera model id {model_id} in {path}. Add to mapping.")
            params = list(_read_next_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[int(cam_id)] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_bin(path: str) -> Dict[int, Dict[str, Any]]:
    images: Dict[int, Dict[str, Any]] = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_id = _read_next_bytes(fid, 4, "I")[0]
            qvec = np.array(_read_next_bytes(fid, 8 * 4, "dddd"), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 8 * 3, "ddd"), dtype=np.float64)
            camera_id = _read_next_bytes(fid, 4, "I")[0]
            # name (null-terminated)
            name_bytes = b""
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            # skip 2D points
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * (8 * 2 + 8), 1)
            images[int(image_id)] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": int(camera_id),
                "name": name,
            }
    return images


def write_cameras_txt(path: str, cams: Dict[int, Dict[str, Any]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cams)}",
    ]
    for cid in sorted(cams.keys()):
        cam = cams[cid]
        params_str = " ".join("{:.17g}".format(float(x)) for x in cam["params"])
        lines.append(f"{cid} {cam['model']} {cam['width']} {cam['height']} {params_str}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_txt(path: str, ims: Dict[int, Dict[str, Any]]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(ims)}",
    ]
    for iid in sorted(ims.keys()):
        im = ims[iid]
        q = np.asarray(im["qvec"], dtype=np.float64).reshape(4)
        t = np.asarray(im["tvec"], dtype=np.float64).reshape(3)
        lines.append(
            "{} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {} {}".format(
                int(iid),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(t[0]), float(t[1]), float(t[2]),
                int(im["camera_id"]),
                str(im["name"]),
            )
        )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_insertion_transform(
    report: dict,
    car_ply: str,
    car_scale: float,
    center_method: str = "density",
    voxel_size: float = 1.0,
    topk_voxels: int = 200,
    clip_q: float = 0.02,
    radius_factor: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (s, R_total, t0, car_center, place)"""
    sim3 = report.get("sim3")
    if isinstance(sim3, dict) and all(k in sim3 for k in ("s", "R", "t")):
        s = float(sim3["s"])
        R_total = np.array(sim3["R"], dtype=np.float64).reshape(3, 3)
        t0 = np.array(sim3["t"], dtype=np.float64).reshape(3)
        car_center = np.array(sim3.get("pivot_car_center", [0, 0, 0]), dtype=np.float64).reshape(3)
        place = report.get("place_final") or report.get("place") or _get(report, ["placement", "place_final"], None)
        if place is None:
            place = (t0 + s * (R_total @ car_center)).tolist()
        place = np.array(place, dtype=np.float64).reshape(3)
        return s, R_total, t0, car_center, place

    car_xyz = load_ply_xyz(car_ply)
    if center_method == "density":
        car_center, _ext, _r, _nsel = dense_center_radius_voxel(
            car_xyz, voxel_size=voxel_size, topk_voxels=topk_voxels, clip_q=clip_q, radius_factor=radius_factor
        )
    elif center_method == "quantile":
        q = float(np.clip(clip_q, 0.0, 0.49))
        lo = np.quantile(car_xyz, q, axis=0)
        hi = np.quantile(car_xyz, 1.0 - q, axis=0)
        car_center = 0.5 * (lo + hi)
    else:
        raise ValueError("center_method must be density or quantile")

    place = report.get("place_final") or report.get("place") or _get(report, ["placement", "place_final"], None)
    if place is None:
        place = _get(report, ["placement", "place_initial"], None)
    if place is None:
        raise ValueError("insert_report must contain place_final/place or placement.place_final")
    place = np.array(place, dtype=np.float64).reshape(3)

    R_total = np.eye(3, dtype=np.float64)
    Rk = report.get("R_total") or _get(report, ["orientation", "R_total"], None)
    if Rk is not None:
        R_total = np.array(Rk, dtype=np.float64).reshape(3, 3)

    s = float(report.get("car_scale", report.get("scale", car_scale)))
    if not np.isfinite(s) or s <= 0:
        s = float(car_scale)

    t0 = place - s * (R_total @ car_center)
    return s, R_total, t0, car_center, place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car_ply", required=True)
    ap.add_argument("--insert_report", required=True)
    ap.add_argument("--colmap_sparse_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--car_scale", type=float, default=1.0)
    ap.add_argument("--center_method", default="density", choices=["density", "quantile"])
    ap.add_argument("--voxel_size", type=float, default=1.0)
    ap.add_argument("--topk_voxels", type=int, default=200)
    ap.add_argument("--clip_q", type=float, default=0.02)
    ap.add_argument("--radius_factor", type=float, default=1.0)
    ap.add_argument("--write_summary", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(Path(args.insert_report).read_text(encoding="utf-8"))
    s, R_total, t0, car_center, place = build_insertion_transform(
        report,
        car_ply=args.car_ply,
        car_scale=float(args.car_scale),
        center_method=args.center_method,
        voxel_size=float(args.voxel_size),
        topk_voxels=int(args.topk_voxels),
        clip_q=float(args.clip_q),
        radius_factor=float(args.radius_factor),
    )

    colmap_dir = Path(args.colmap_sparse_dir)
    cams = read_cameras_bin(str(colmap_dir / "cameras.bin"))
    ims = read_images_bin(str(colmap_dir / "images.bin"))

    ims_scene: Dict[int, Dict[str, Any]] = {}
    for iid, im in ims.items():
        Rcw = qvec2rotmat(im["qvec"])
        t = np.asarray(im["tvec"], dtype=np.float64).reshape(3)

        # camera center in car/COLMAP world
        Rwc = Rcw.T
        C = -Rwc @ t

        # transform center into scene world
        C2 = (s * (R_total @ C)) + t0

        # rotate world frame
        Rwc2 = (R_total @ Rwc)
        Rcw2 = Rwc2.T
        t2 = -Rcw2 @ C2

        ims_scene[int(iid)] = {
            "qvec": rotmat2qvec_wxyz(Rcw2),
            "tvec": t2,
            "camera_id": int(im["camera_id"]),
            "name": str(im["name"]),
        }

    write_cameras_txt(str(out_dir / "cameras_scene.txt"), cams)
    write_images_txt(str(out_dir / "images_scene.txt"), ims_scene)

    if args.write_summary:
        summary = {
            "car_scale": float(s),
            "R_total": R_total.tolist(),
            "t0": t0.tolist(),
            "car_center": car_center.tolist(),
            "place": place.tolist(),
            "n_images": int(len(ims_scene)),
            "n_cameras": int(len(cams)),
        }
        (out_dir / "transform_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print("Wrote:")
    print(str(out_dir / "cameras_scene.txt"))
    print(str(out_dir / "images_scene.txt"))
    if args.write_summary:
        print(str(out_dir / "transform_summary.json"))


if __name__ == "__main__":
    main()
