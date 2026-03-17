#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify custom/threestudio-3dgs/utils module by rendering COLMAP TXT cameras.

This script is intentionally close to your validated v29 renderer:
- Uses the same "ts3dgs" synthetic namespace import mechanism (hyphenated folder safe).
- Selectively registers required threestudio modules: geometry "gaussian-splatting" and renderer "diff-gaussian-rasterizer".
- Loads and patches geometry/renderer as in v29.
- Builds viewpoint cameras via YOUR utils module:
    ts3dgs.utils.gs_camera_from_colmap.build_viewpoints_from_txt(...)
- Renders each view and writes PNGs.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ----------------------------
# v29-style import helpers
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
# pc API patch (copied from v29; made robust)
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
                if torch.is_tensor(f) and f.dim() == 2:
                    return f
                if torch.is_tensor(f):
                    return f.view(f.shape[0], -1, f.shape[-1])
            raise AttributeError("Cannot find SH features.")
        setattr(cls, "get_features", get_features)

    return pc


# ----------------------------
# Renderer forward patch (v29 style; fixed)
# ----------------------------

def patch_renderer_forward(renderer):
    import types
    import math

    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except Exception:
        # If extension is unavailable, keep renderer as-is.
        return renderer

    def _pick_first(*candidates):
        for x in candidates:
            if x is None:
                continue
            return x
        return None

    def _maybe_call(x):
        try:
            return x() if callable(x) else x
        except TypeError:
            return x

    def forward_patched(self, viewpoint_camera, bg, scaling_modifier=1.0, override_color=None, **kwargs):
        dev = bg.device if torch.is_tensor(bg) else torch.device("cuda")

        pc = (
            getattr(self, "pc", None)
            or getattr(self, "gaussians", None)
            or kwargs.get("geometry", None)
            or kwargs.get("gaussians", None)
            or kwargs.get("pc", None)
        )
        if pc is None:
            raise ValueError("Renderer has no geometry bound (pc/gaussians).")

        def _get_pc(name: str):
            if hasattr(pc, name):
                return _maybe_call(getattr(pc, name))
            return None

        xyz_for_ss = _pick_first(_get_pc("get_xyz"), _get_pc("xyz"), getattr(pc, "_xyz", None))
        if xyz_for_ss is None:
            raise AttributeError("Geometry has no xyz (tried get_xyz/xyz/_xyz)")
        screenspace_points = torch.zeros_like(xyz_for_ss, requires_grad=True, device=dev) + 0

        tanfovx = math.tan(float(viewpoint_camera.FoVx) * 0.5)
        tanfovy = math.tan(float(viewpoint_camera.FoVy) * 0.5)

        proj_raw = getattr(viewpoint_camera, "projection_matrix", None)
        if proj_raw is None:
            proj_raw = viewpoint_camera.full_proj_transform

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg.to(device=dev) if torch.is_tensor(bg) else torch.tensor(bg, device=dev, dtype=torch.float32),
            scale_modifier=float(scaling_modifier),
            viewmatrix=viewpoint_camera.world_view_transform.to(device=dev),
            projmatrix=viewpoint_camera.full_proj_transform.to(device=dev),
            projmatrix_raw=proj_raw,
            sh_degree=int(getattr(pc, "active_sh_degree", 3)),
            campos=viewpoint_camera.camera_center.to(device=dev),
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = _pick_first(_get_pc("get_xyz"), _get_pc("xyz"), getattr(pc, "_xyz", None))
        if means3D is None:
            raise AttributeError("Geometry has no xyz (tried get_xyz/xyz/_xyz)")
        means2D = screenspace_points

        opacity = _pick_first(_get_pc("get_opacity"), _get_pc("opacity"), getattr(pc, "_opacity", None))
        scales = _pick_first(_get_pc("get_scaling"), _get_pc("scales"), getattr(pc, "_scaling", None))
        rotations = _pick_first(_get_pc("get_rotation"), _get_pc("rotation"), getattr(pc, "_rotation", None))

        if opacity is None:
            raise AttributeError("Geometry has no opacity (tried get_opacity/opacity/_opacity)")
        if scales is None:
            raise AttributeError("Geometry has no scales (tried get_scaling/scales/_scaling)")
        if rotations is None:
            raise AttributeError("Geometry has no rotations (tried get_rotation/rotation/_rotation)")

        shs = None
        colors_precomp = None
        if override_color is None:
            shs = _pick_first(_get_pc("get_features"), _get_pc("features"))
            if shs is None and hasattr(pc, "_features_dc") and hasattr(pc, "_features_rest"):
                shs = torch.cat([pc._features_dc, pc._features_rest], dim=1)
            if shs is None:
                raise AttributeError(
                    "Geometry has no SH/features (tried get_features/features/_features_dc+_features_rest)"
                )
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
# Instantiate geometry + renderer
# ----------------------------

def build_geometry(ply_path: str, sh_degree: int, device: torch.device):
    import threestudio
    GeoCls = threestudio.find("gaussian-splatting")
    geo = GeoCls({
        "sh_degree": int(sh_degree),
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
# Load utils module by file path into ts3dgs.utils.*
# ----------------------------

def load_ts3dgs_utils(repo_root: str):
    root = Path(repo_root) / "custom" / "threestudio-3dgs"
    utils = root / "utils"
    if not utils.exists():
        raise FileNotFoundError(f"Cannot find utils folder: {utils}")

    _mk_pkg("ts3dgs", root)
    _mk_pkg("ts3dgs.utils", utils)

    colmap_txt = utils / "colmap_txt_io.py"
    cam_mod = utils / "gs_camera_from_colmap.py"
    if not colmap_txt.exists() or not cam_mod.exists():
        raise FileNotFoundError("utils module files missing. Expected colmap_txt_io.py and gs_camera_from_colmap.py")

    import_module_from_file("ts3dgs.utils.colmap_txt_io", colmap_txt)
    m = import_module_from_file("ts3dgs.utils.gs_camera_from_colmap", cam_mod)
    return m


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

    bg_rgb = tuple(float(x) for x in args.bg.split(","))
    if len(bg_rgb) != 3:
        raise ValueError("--bg must be r,g,b")

    register_needed_modules(args.repo_root)

    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    utils_mod = load_ts3dgs_utils(args.repo_root)
    build_viewpoints_from_txt = utils_mod.build_viewpoints_from_txt

    geometry = patch_pc_api(build_geometry(args.combined_ply, args.sh_degree, dev))
    renderer = patch_renderer_forward(build_renderer(geometry, bg_rgb, dev))
    bg_tensor = torch.tensor(bg_rgb, dtype=torch.float32, device=dev)

    views = build_viewpoints_from_txt(
        repo_root=args.repo_root,
        cameras_txt=args.cameras_txt,
        images_txt=args.images_txt,
        device=dev,
        render_width=args.render_width,
        render_height=args.render_height,
        znear=args.znear,
        zfar=args.zfar,
        max_images=args.max_images,
    )

    from PIL import Image

    for idx, (view_cam, meta) in enumerate(views):
        with torch.no_grad():
            out = renderer.forward(view_cam, bg_tensor)
            rgb = out["render"]  # [3,H,W]
        rgb_np = (rgb.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        base = Path(getattr(meta, "name", f"img_{idx:05d}")).stem
        out_name = f"{base}.png"
        Image.fromarray(rgb_np).save(str(Path(args.out_dir) / out_name))

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(views)}] rendered {out_name}")

    print("DONE")
    print(f"Wrote renders to: {args.out_dir}")


if __name__ == "__main__":
    main()
