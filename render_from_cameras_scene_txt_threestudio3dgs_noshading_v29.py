#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render from cameras_scene.txt / images_scene.txt using threestudio-3dgs *non-shading* renderer.

Based on: render_from_cameras_scene_txt_threestudio3dgs_noshading_v20.py (your uploaded working base).
Fix: car/object being "squashed" (aspect distortion).

Root cause:
- GaussianBatchRenderer.batch_forward in this fork uses `fovy` for BOTH FoVx and FoVy,
  which is only correct for square images and fx==fy. For rectangular images, it distorts aspect.
- Your v20 script computed fx/fy but still went through batch_forward, so it got squashed.

Fix implemented here:
- Compute fovx from fx+W and fovy from fy+H (in radians)
- Build gsrender-style Camera with FoVx=fovx, FoVy=fovy using threestudio.utils.ops.get_cam_info_gaussian
- Call renderer.forward(viewpoint_camera, bg_color) directly (bypassing batch_forward)

Everything else (module registration, projmatrix_raw patch, SH robustness) stays aligned with v20.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Any, Optional

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
    # radians
    return float(2.0 * math.atan(float(w) / (2.0 * float(fx) + 1e-9)))


COLMAP_TO_OPENGL = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float32)


def w2c_to_c2w(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    Rcw = np.asarray(Rcw, dtype=np.float64)
    tcw = np.asarray(tcw, dtype=np.float64).reshape(3)
    Rwc = Rcw.T
    C = (-Rwc @ tcw.reshape(3, 1)).reshape(3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rwc.astype(np.float32)
    c2w[:3, 3] = C.astype(np.float32)
    return c2w


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


# ----------------------------
# Selective register (minimal: geometry + diff-gaussian-rasterizer)
# (Copied from v20)
# ----------------------------

def ensure_repo_on_path(repo_root: str):
    import sys
    if repo_root and os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _mk_pkg(name: str, path: Path):
    import sys, types
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(path)]  # type: ignore[attr-defined]
    m.__package__ = name
    sys.modules[name] = m


def import_module_from_file(qualified_name: str, file_path: Path):
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location(qualified_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {qualified_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def find_register_file(folder: Path, key: str) -> Optional[Path]:
    pat1 = f'register("{key}")'
    pat2 = f"register('{key}')"
    for py in sorted(folder.glob("*.py")):
        if py.name == "__init__.py":
            continue
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pat1 in txt or pat2 in txt:
            return py
    return None


def ensure_registered(repo_root: str, sub: str, key: str):
    import threestudio
    try:
        _ = threestudio.find(key)
        return
    except KeyError:
        pass

    root = Path(repo_root) / "custom" / "threestudio-3dgs"
    _mk_pkg("ts3dgs", root)
    folder = root / sub
    _mk_pkg(f"ts3dgs.{sub}", folder)

    py = find_register_file(folder, key)
    if py is None:
        raise FileNotFoundError(f"Cannot find module registering '{key}' under {folder}")

    import_module_from_file(f"ts3dgs.{sub}.{py.stem}", py)


def register_needed_modules(repo_root: str):
    ensure_repo_on_path(repo_root)
    ensure_registered(repo_root, "geometry", "gaussian-splatting")
    ensure_registered(repo_root, "renderer", "diff-gaussian-rasterizer")


# ----------------------------
# pc API patch (Copied from v20)
# ----------------------------

def patch_pc_api(pc):
    cls = pc.__class__

    def _has(name: str) -> bool:
        return hasattr(cls, name) or hasattr(pc, name)

    if not hasattr(pc, "active_sh_degree"):
        shd = getattr(getattr(pc, "cfg", None), "sh_degree", None)
        if shd is None:
            shd = getattr(pc, "max_sh_degree", 3)
        try:
            setattr(pc, "active_sh_degree", int(shd))
        except Exception:
            pass

    if not _has("get_xyz"):
        if hasattr(pc, "_xyz"):
            @property
            def get_xyz(self):
                return self._xyz
            setattr(cls, "get_xyz", get_xyz)
        elif hasattr(pc, "xyz"):
            @property
            def get_xyz(self):
                return self.xyz
            setattr(cls, "get_xyz", get_xyz)

    if not _has("get_scaling"):
        if hasattr(pc, "_scaling") and hasattr(pc, "scaling_activation"):
            @property
            def get_scaling(self):
                return self.scaling_activation(self._scaling)
            setattr(cls, "get_scaling", get_scaling)
        elif hasattr(pc, "scaling") and hasattr(pc, "scaling_activation"):
            @property
            def get_scaling(self):
                return self.scaling_activation(self.scaling)
            setattr(cls, "get_scaling", get_scaling)

    if not _has("get_rotation"):
        if hasattr(pc, "_rotation") and hasattr(pc, "rotation_activation"):
            @property
            def get_rotation(self):
                return self.rotation_activation(self._rotation)
            setattr(cls, "get_rotation", get_rotation)
        elif hasattr(pc, "rotation") and hasattr(pc, "rotation_activation"):
            @property
            def get_rotation(self):
                return self.rotation_activation(self.rotation)
            setattr(cls, "get_rotation", get_rotation)

    if not _has("get_opacity"):
        if hasattr(pc, "_opacity") and hasattr(pc, "opacity_activation"):
            @property
            def get_opacity(self):
                return self.opacity_activation(self._opacity)
            setattr(cls, "get_opacity", get_opacity)
        elif hasattr(pc, "opacity") and hasattr(pc, "opacity_activation"):
            @property
            def get_opacity(self):
                return self.opacity_activation(self.opacity)
            setattr(cls, "get_opacity", get_opacity)

    if not _has("get_features"):

        @property
        def get_features(self):
            if hasattr(self, "_features_dc") and hasattr(self, "_features_rest"):
                return torch.cat((self._features_dc, self._features_rest), dim=1)
            if hasattr(self, "features_dc") and hasattr(self, "features_rest"):
                return torch.cat((self.features_dc, self.features_rest), dim=1)
            if hasattr(self, "features"):
                f = self.features
                if torch.is_tensor(f):
                    if f.ndim == 3:
                        if f.shape[-1] == 3:
                            return f
                        if f.shape[1] == 3:
                            return f.transpose(1, 2).contiguous()
                    return f
            if hasattr(self, "shs"):
                return self.shs
            raise AttributeError(
                "Cannot build get_features: expected (_features_dc/_features_rest) or "
                "(features_dc/features_rest) or (features/shs)."
            )

        setattr(cls, "get_features", get_features)

    return pc


# ----------------------------
# Patch renderer.forward to pass projmatrix_raw (Copied from v20)
# ----------------------------

def patch_renderer_forward_add_projraw(renderer, znear: float, zfar: float):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    import types
    import numpy as _np
    import math as _math

    def forward_patched(self, viewpoint_camera, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, **kwargs):
        if self.training:
            invert_bg_color = _np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True
        bg = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry
        dev = pc.get_xyz.device

        screenspace_points = (
            torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=dev) + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = _math.tan(float(viewpoint_camera.FoVx) * 0.5)
        tanfovy = _math.tan(float(viewpoint_camera.FoVy) * 0.5)

        P = get_projection_matrix(float(znear), float(zfar), float(viewpoint_camera.FoVx), float(viewpoint_camera.FoVy))
        proj_raw = torch.from_numpy(P).to(device=dev, dtype=torch.float32).transpose(0, 1)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg.to(device=dev),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(device=dev),
            projmatrix=viewpoint_camera.full_proj_transform.to(device=dev),
            projmatrix_raw=proj_raw,
            sh_degree=int(getattr(pc, "active_sh_degree", 3)),
            campos=viewpoint_camera.camera_center.to(device=dev),
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation

        shs = None
        colors_precomp = None
        if override_color is None:
            if hasattr(pc, "get_features"):
                shs = pc.get_features
            elif hasattr(pc, "_features_dc") and hasattr(pc, "_features_rest"):
                shs = torch.cat((pc._features_dc, pc._features_rest), dim=1)
            elif hasattr(pc, "features_dc") and hasattr(pc, "features_rest"):
                shs = torch.cat((pc.features_dc, pc.features_rest), dim=1)
            else:
                raise AttributeError("Cannot find SH features on geometry.")
        else:
            colors_precomp = override_color

        raster_out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        if isinstance(raster_out, (tuple, list)):
            rendered_image = raster_out[0]
            radii = raster_out[1] if len(raster_out) > 1 else None
        elif isinstance(raster_out, dict):
            rendered_image = raster_out.get("render", raster_out.get("rendered_image"))
            radii = raster_out.get("radii")
        else:
            rendered_image = raster_out
            radii = None

        return {
            "render": rendered_image.clamp(0, 1),
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0) if radii is not None else None,
            "radii": radii,
        }

    renderer.forward = types.MethodType(forward_patched, renderer)
    return renderer


# ----------------------------
# Instantiate geometry + renderer (Copied from v20)
# ----------------------------

def build_geometry(ply_path: str, sh_degree: int, device: torch.device):
    import threestudio
    GeoCls = threestudio.find("gaussian-splatting")
    geo = GeoCls({
        "sh_degree": sh_degree,
        "geometry_convert_from": ply_path,
        "load_ply_only_vertex": False,
        "init_num_pts": 100,
        "pc_init_radius": 0.8,
        "opacity_init": 0.1,
    })
    geo.to(device=device.type)
    return geo


def build_renderer(geometry, bg_rgb, device: torch.device):
    import threestudio
    RenCls = threestudio.find("diff-gaussian-rasterizer")

    from threestudio.models.materials.base import BaseMaterial
    from threestudio.models.background.base import BaseBackground

    class _DummyMat(BaseMaterial):
        def forward(self, *args, **kwargs):
            return {}

    class _DummyBg(BaseBackground):
        def forward(self, *args, **kwargs):
            return {}

    dummy_mat = _DummyMat({})
    dummy_bg = _DummyBg({})

    renderer = RenCls({"debug": False, "back_ground_color": bg_rgb}, geometry=geometry, material=dummy_mat, background=dummy_bg)
    if hasattr(renderer, "background_tensor"):
        renderer.background_tensor = torch.tensor(bg_rgb, dtype=torch.float32, device=device)
    for attr in ["pc", "gaussians"]:
        try:
            setattr(renderer, attr, geometry)
        except Exception:
            pass
    return renderer


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/nfs4/qhy/projects/threestudio")
    ap.add_argument("--combined_ply", type=str, required=True)
    ap.add_argument("--cameras_txt", type=str, required=True)
    ap.add_argument("--images_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--render_width", type=int, default=0, help="0=use camera width; else render width.")
    ap.add_argument("--render_height", type=int, default=0, help="0=use camera height; else render height.")
    ap.add_argument("--bg", type=str, default="1,1,1")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--znear", type=float, default=0.1)
    ap.add_argument("--zfar", type=float, default=100.0)
    args = ap.parse_args()

    bg = tuple(float(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg must be r,g,b in 0..1")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    register_needed_modules(args.repo_root)

    cams = read_cameras_txt(args.cameras_txt)
    ims = read_images_txt(args.images_txt)
    if len(cams) == 0 or len(ims) == 0:
        raise ValueError("Parsed empty cameras/images. Check your *_scene.txt paths.")

    geometry = build_geometry(args.combined_ply, sh_degree=args.sh_degree, device=device)
    patch_pc_api(geometry)

    renderer = build_renderer(geometry=geometry, bg_rgb=bg, device=device)
    patch_renderer_forward_add_projraw(renderer, znear=args.znear, zfar=args.zfar)

    # Import Camera + helper to build matrices (works because v20 created ts3dgs namespace pkgs)
    import importlib
    from threestudio.utils.ops import get_cam_info_gaussian
    gaussian_base_mod = importlib.import_module("ts3dgs.geometry.gaussian_base")
    Camera = gaussian_base_mod.Camera

    from PIL import Image

    image_ids = sorted(ims.keys())
    if args.max_images and args.max_images > 0:
        image_ids = image_ids[:args.max_images]

    used = set()
    for idx, iid in enumerate(image_ids):
        im = ims[iid]
        cam = cams[int(im["camera_id"])]

        W_cam = int(cam["width"])
        H_cam = int(cam["height"])
        W = int(args.render_width) if int(args.render_width) > 0 else W_cam
        H = int(args.render_height) if int(args.render_height) > 0 else H_cam

        model = cam["model"].upper()
        params = cam["params"]

        # fx/fy (distortion ignored)
        if model == "PINHOLE":
            fx, fy = float(params[0]), float(params[1])
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = float(params[0]); fy = fx
        elif model in ("OPENCV", "OPENCV_FISHEYE"):
            fx, fy = float(params[0]), float(params[1])
        else:
            fx = float(params[0]); fy = fx

        # ✅ aspect-correct FoVs
        fovx = fx_to_fov(fx, W_cam)  # use camera's native intrinsics size
        fovy = fx_to_fov(fy, H_cam)

        # pose -> c2w (scene coords already)
        Rcw = qvec2rotmat_wxyz(im["qvec"])
        tcw = np.asarray(im["tvec"], dtype=np.float64)
        c2w_np = w2c_to_c2w(Rcw, tcw)

        # COLMAP -> OpenGL convention
        c2w_np = c2w_np @ COLMAP_TO_OPENGL

        c2w = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)

        w2c, full_proj, cam_center = get_cam_info_gaussian(
            c2w=c2w,
            fovx=torch.tensor(fovx, device=device, dtype=torch.float32),
            fovy=torch.tensor(fovy, device=device, dtype=torch.float32),
            znear=float(args.znear),
            zfar=float(args.zfar),
        )

        viewpoint = Camera(
            torch.tensor(fovx, device=device, dtype=torch.float32),
            torch.tensor(fovy, device=device, dtype=torch.float32),
            cam_center,
            W,
            H,
            w2c,
            full_proj,
        )

        with torch.no_grad():
            render_pkg = renderer.forward(viewpoint, renderer.background_tensor)
            rgb_chw = render_pkg["render"]  # [3,H,W]
            rgb = rgb_chw.permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()

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
