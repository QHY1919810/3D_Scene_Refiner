from __future__ import annotations

"""
Build gsrender-style Camera objects from COLMAP TXT entries.

IMPORTANT (matches your working v29 script):
- Your fork lives under a folder named "custom/threestudio-3dgs" (with a hyphen),
  which is NOT importable as a normal Python package path.
- The working approach (v29) is to create a synthetic package namespace "ts3dgs"
  and then import modules from file paths into that namespace.
  We replicate that approach here to reliably access:
    - ts3dgs.geometry.gaussian_base.Camera   (NamedTuple expected by renderer.forward)
    - and to keep the Camera definition exactly aligned with the fork.
- FoV is computed aspect-correctly:
    fovx from fx + W_cam, fovy from fy + H_cam (in radians)
  so rectangular images do not get squashed.

This module is intended to be the single source of truth for:
- parsing COLMAP qvec/tvec (world->cam)
- converting to c2w (cam->world) in scene coordinates
- converting COLMAP convention to OpenGL-like convention used by the 3DGS renderer (COLMAP_TO_OPENGL)
- building Camera using threestudio.utils.ops.get_cam_info_gaussian
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

from .colmap_txt_io import ColmapCamera, ColmapImage

import sys
from typing import NamedTuple
from pathlib import Path

# 目标文件名（用于 sys.modules 扫描定位）
_GB_BASENAME = "gaussian_base.py"


# ---------------------------------------------------------------------
# COLMAP pose helpers
# ---------------------------------------------------------------------

def qvec2rotmat_wxyz(q):
    """q = [w,x,y,z] -> Rcw (world->cam)."""
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    R = np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ],
        dtype=np.float64,
    )
    return R


def w2c_to_c2w(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    """Rcw/tcw (world->cam) -> c2w (cam->world) 4x4."""
    Rcw = np.asarray(Rcw, dtype=np.float64)
    tcw = np.asarray(tcw, dtype=np.float64).reshape(3)
    Rwc = Rcw.T
    C = (-Rwc @ tcw.reshape(3, 1)).reshape(3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rwc.astype(np.float32)
    c2w[:3, 3] = C.astype(np.float32)
    return c2w


def fx_to_fov(fx: float, w: int) -> float:
    """Return FoV in radians."""
    return float(2.0 * math.atan(float(w) / (2.0 * float(fx) + 1e-9)))


# COLMAP (OpenCV-like) -> OpenGL-like convention used by gsrender/3DGS
COLMAP_TO_OPENGL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def get_fx_fy(cam: ColmapCamera) -> Tuple[float, float]:
    """Distortion ignored (same behavior as your v29 debug renderer)."""
    model = cam.model.upper()
    p = cam.params
    if model == "PINHOLE":
        return float(p[0]), float(p[1])  # fx, fy, cx, cy
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = float(p[0])
        return fx, fx
    if model in ("OPENCV", "OPENCV_FISHEYE"):
        return float(p[0]), float(p[1])
    fx = float(p[0])
    return fx, fx


# ---------------------------------------------------------------------
# v29-style module import (ts3dgs namespace)
# ---------------------------------------------------------------------

def _ensure_repo_on_path(repo_root: str) -> None:
    import sys

    if repo_root and os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _mk_pkg(name: str, path: Path) -> None:
    import sys, types

    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(path)]  # type: ignore[attr-defined]
    m.__package__ = name
    sys.modules[name] = m


def _import_module_from_file(qualified_name: str, file_path: Path):
    import sys, importlib.util

    spec = importlib.util.spec_from_file_location(qualified_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {qualified_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _get_gaussian_base_module(repo_root: str):
    """Load gaussian_base.py into a synthetic namespace: ts3dgs.geometry.gaussian_base"""
    repo_root = str(repo_root)
    _ensure_repo_on_path(repo_root)

    root = Path(repo_root) / "custom" / "threestudio-3dgs"
    geom = root / "geometry"
    gb_py = geom / "gaussian_base.py"
    if not gb_py.exists():
        raise FileNotFoundError(f"Cannot find gaussian_base.py at {gb_py}")

    _mk_pkg("ts3dgs", root)
    _mk_pkg("ts3dgs.geometry", geom)

    return _import_module_from_file("ts3dgs.geometry.gaussian_base", gb_py)


def _find_camera_in_sysmodules():
    """
    Return Camera NamedTuple if gaussian_base.py has already been imported by threestudio.
    This avoids re-executing gaussian_base.py (which would re-register extensions and crash).
    """
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        if Path(f).name != _GB_BASENAME:
            continue
        if hasattr(mod, "Camera"):
            return getattr(mod, "Camera")
    return None


def _define_fallback_camera_namedtuple():
    """
    Fallback: define a Camera NamedTuple with the exact fields your renderer expects.
    This has no side effects and is safe even if gaussian_base isn't importable.
    """
    class Camera(NamedTuple):
        FoVx: torch.Tensor
        FoVy: torch.Tensor
        camera_center: torch.Tensor
        image_width: int
        image_height: int
        world_view_transform: torch.Tensor
        full_proj_transform: torch.Tensor

    return Camera


def get_camera_namedtuple(repo_root: str = ""):
    """
    Get the fork's Camera type WITHOUT re-importing gaussian_base.py.
    """
    cam = _find_camera_in_sysmodules()
    if cam is not None:
        return cam

    # last resort: define locally
    return _define_fallback_camera_namedtuple()

# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------

@dataclass
class CameraMeta:
    image_id: int
    camera_id: int
    name: str
    fovx: float
    fovy: float
    W_cam: int
    H_cam: int
    W_render: int
    H_render: int


def build_camera_from_colmap(
    repo_root: str,
    cam: ColmapCamera,
    img: ColmapImage,
    device: Union[str, torch.device] = "cuda",
    render_width: int = 0,
    render_height: int = 0,
    znear: float = 0.1,
    zfar: float = 100.0,
):
    """Build fork Camera for renderer.forward (v29-equivalent)."""
    from threestudio.utils.ops import get_cam_info_gaussian

    dev = torch.device(device) if isinstance(device, str) else device
    Camera = get_camera_namedtuple(repo_root)

    W_cam, H_cam = int(cam.width), int(cam.height)
    W = int(render_width) if int(render_width) > 0 else W_cam
    H = int(render_height) if int(render_height) > 0 else H_cam

    fx, fy = get_fx_fy(cam)
    fovx = fx_to_fov(fx, W_cam)
    fovy = fx_to_fov(fy, H_cam)

    Rcw = qvec2rotmat_wxyz(img.qvec_wxyz)
    tcw = np.asarray(img.tvec, dtype=np.float64)
    c2w_np = w2c_to_c2w(Rcw, tcw)
    c2w_np = c2w_np @ COLMAP_TO_OPENGL

    c2w = torch.from_numpy(c2w_np).to(device=dev, dtype=torch.float32)

    w2c, full_proj, cam_center = get_cam_info_gaussian(
        c2w=c2w,
        fovx=torch.tensor(fovx, device=dev, dtype=torch.float32),
        fovy=torch.tensor(fovy, device=dev, dtype=torch.float32),
        znear=float(znear),
        zfar=float(zfar),
    )

    viewpoint = Camera(
        torch.tensor(fovx, device=dev, dtype=torch.float32),
        torch.tensor(fovy, device=dev, dtype=torch.float32),
        cam_center,
        W,
        H,
        w2c,
        full_proj,
    )

    meta = CameraMeta(
        image_id=int(img.id),
        camera_id=int(img.camera_id),
        name=str(img.name),
        fovx=float(fovx),
        fovy=float(fovy),
        W_cam=W_cam,
        H_cam=H_cam,
        W_render=W,
        H_render=H,
    )
    return viewpoint, meta


def build_viewpoints_from_txt(
    repo_root: str,
    cameras_txt: str,
    images_txt: str,
    device: Union[str, torch.device] = "cuda",
    render_width: int = 0,
    render_height: int = 0,
    znear: float = 0.1,
    zfar: float = 100.0,
    max_images: int = 0,
):
    """Read TXT + build a list of (Camera, meta) sorted by image_id."""
    from .colmap_txt_io import read_cameras_txt, read_images_txt

    cams = read_cameras_txt(cameras_txt)
    ims = read_images_txt(images_txt)

    image_ids = sorted(ims.keys())
    if max_images and max_images > 0:
        image_ids = image_ids[:max_images]

    out = []
    for iid in image_ids:
        img = ims[iid]
        cam = cams[int(img.camera_id)]
        out.append(
            build_camera_from_colmap(
                repo_root=repo_root,
                cam=cam,
                img=img,
                device=device,
                render_width=render_width,
                render_height=render_height,
                znear=znear,
                zfar=zfar,
            )
        )
    return out
