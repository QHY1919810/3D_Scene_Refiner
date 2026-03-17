#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test pipeline: observed-camera render (threestudio-3dgs) + Qwen-Image-Edit refinement (DiffSynth).

What it does:
1) Loads your supervised COLMAP-scene datamodule (colmap-scene-datamodule).
2) For N samples (original COLMAP viewpoints):
   - renders the current 3DGS PLY with threestudio-3dgs renderer at that exact viewpoint
   - runs DiffSynth Qwen-Image-Edit-2511 to "polish" the rendered image (no gradients)
   - writes BOTH images:
       out_dir/render/<view_id>_<name>.png
       out_dir/qwen_edit/<view_id>_<name>.png

This script does NOT use guidance/loss/backprop; it's purely an I/O sanity check.

Run inside your threestudio environment.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ----------------------------
# Minimal module registration helpers (same pattern as your v29 render script)
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
    ensure_registered(repo_root, "data", "colmap-scene-datamodule")


# ----------------------------
# Patches copied from v29 (robustness)
# ----------------------------
def patch_pc_api(pc):
    """
    One-shot, render-only patch:
    - Removes class-level 'get_features' descriptor if it exists (from previous patches).
    - Injects required 'get_*' tensors into instance __dict__ to bypass nn.Module.__setattr__/__getattr__.

    This makes diff_gaussian_rasterizer.py happy for rendering.
    """

    import torch

    cls = pc.__class__

    # ------------------------------------------------------------
    # 0) If someone previously attached @property get_features to the class,
    #    delete it so instance injection is possible.
    # ------------------------------------------------------------
    try:
        if "get_features" in getattr(cls, "__dict__", {}) and isinstance(cls.__dict__["get_features"], property):
            delattr(cls, "get_features")
    except Exception:
        pass

    # ------------------------------------------------------------
    # 1) active_sh_degree / max_sh_degree
    # ------------------------------------------------------------
    if not hasattr(pc, "active_sh_degree"):
        shd = getattr(getattr(pc, "cfg", None), "sh_degree", None)
        if shd is None:
            shd = getattr(pc, "max_sh_degree", None)
        if shd is None:
            shd = 3
        try:
            object.__setattr__(pc, "active_sh_degree", int(shd))
        except Exception:
            pc.__dict__["active_sh_degree"] = int(shd)

    if not hasattr(pc, "max_sh_degree"):
        shd = getattr(getattr(pc, "cfg", None), "sh_degree", None)
        if shd is None:
            shd = 3
        try:
            object.__setattr__(pc, "max_sh_degree", int(shd))
        except Exception:
            pc.__dict__["max_sh_degree"] = int(shd)

    # ------------------------------------------------------------
    # 2) Inject required tensors into instance dict
    #    These are the names rasterizer reads as attributes:
    #      pc.get_xyz, pc.get_scaling, pc.get_rotation, pc.get_opacity, pc.get_features
    # ------------------------------------------------------------

    # xyz
    if "get_xyz" not in pc.__dict__:
        if hasattr(pc, "_xyz"):
            pc.__dict__["get_xyz"] = pc._xyz
        elif hasattr(pc, "xyz"):
            pc.__dict__["get_xyz"] = pc.xyz

    # scaling (activated)
    if "get_scaling" not in pc.__dict__:
        if hasattr(pc, "_scaling") and hasattr(pc, "scaling_activation"):
            pc.__dict__["get_scaling"] = pc.scaling_activation(pc._scaling)
        elif hasattr(pc, "scaling"):
            pc.__dict__["get_scaling"] = pc.scaling

    # rotation (activated)
    if "get_rotation" not in pc.__dict__:
        if hasattr(pc, "_rotation") and hasattr(pc, "rotation_activation"):
            pc.__dict__["get_rotation"] = pc.rotation_activation(pc._rotation)
        elif hasattr(pc, "rotation"):
            pc.__dict__["get_rotation"] = pc.rotation

    # opacity (activated)
    if "get_opacity" not in pc.__dict__:
        if hasattr(pc, "_opacity") and hasattr(pc, "opacity_activation"):
            pc.__dict__["get_opacity"] = pc.opacity_activation(pc._opacity)
        elif hasattr(pc, "opacity"):
            pc.__dict__["get_opacity"] = pc.opacity

    # SH features (THIS is the one you卡住的)
    if "get_features" not in pc.__dict__:
        if hasattr(pc, "_features_dc") and hasattr(pc, "_features_rest"):
            pc.__dict__["get_features"] = torch.cat([pc._features_dc, pc._features_rest], dim=1)
        elif hasattr(pc, "_features"):
            pc.__dict__["get_features"] = pc._features
        elif hasattr(pc, "features"):
            pc.__dict__["get_features"] = pc.features

    # Optional: expose dc/rest too (some forks use them)
    if "get_features_dc" not in pc.__dict__ and hasattr(pc, "_features_dc"):
        pc.__dict__["get_features_dc"] = pc._features_dc
    if "get_features_rest" not in pc.__dict__ and hasattr(pc, "_features_rest"):
        pc.__dict__["get_features_rest"] = pc._features_rest

    # ------------------------------------------------------------
    # 3) Validate by direct dict access (avoid __getattr__ surprises)
    # ------------------------------------------------------------
    required = ["get_xyz", "get_scaling", "get_rotation", "get_opacity", "get_features"]
    missing = [k for k in required if k not in pc.__dict__]
    if missing:
        raise AttributeError(
            f"patch_pc_api: still missing {missing}. "
            f"Have _features_dc={hasattr(pc,'_features_dc')} _features_rest={hasattr(pc,'_features_rest')}"
        )

# def patch_renderer_forward_add_projraw(renderer, znear: float, zfar: float):
#     """Patch renderer.forward to pass projmatrix_raw if needed (some forks expect it)."""
#     import types
#     from threestudio.utils.ops import get_projection_matrix

#     if not hasattr(renderer, "forward"):
#         return

#     orig_forward = renderer.forward

#     def forward_patched(self, viewpoint_camera, bg_color, **kwargs):
#         # Some rasterizers use a raw projection matrix separately.
#         # If renderer supports it, provide projmatrix_raw.
#         try:
#             proj_raw = get_projection_matrix(
#                 znear=znear,
#                 zfar=zfar,
#                 fovX=float(viewpoint_camera.FoVx),
#                 fovY=float(viewpoint_camera.FoVy),
#             )
#             kwargs.setdefault("projmatrix_raw", proj_raw)
#         except Exception:
#             pass
#         return orig_forward(viewpoint_camera, bg_color, **kwargs)

#     renderer.forward = types.MethodType(forward_patched, renderer)

def patch_renderer_forward_add_projraw(renderer, znear: float, zfar: float):
    import types
    import torch
    import math

    orig_forward = renderer.forward

    def _proj_raw_from_fov(fovx, fovy, device):
        tanfovx = math.tan(float(fovx) * 0.5)
        tanfovy = math.tan(float(fovy) * 0.5)
        P = torch.zeros((4, 4), device=device, dtype=torch.float32)
        P[0, 0] = 1.0 / (tanfovx + 1e-9)
        P[1, 1] = 1.0 / (tanfovy + 1e-9)
        P[2, 2] = float(zfar) / (float(zfar) - float(znear) + 1e-9)
        P[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear) + 1e-9)
        P[3, 2] = 1.0
        return P

    def forward_patched(self, viewpoint_camera, bg_color, **kwargs):
        # 强制生成并注入 projmatrix_raw（不使用 setdefault）
        dev = bg_color.device if isinstance(bg_color, torch.Tensor) else torch.device("cuda")
        P = _proj_raw_from_fov(viewpoint_camera.FoVx, viewpoint_camera.FoVy, dev).transpose(0, 1)
        kwargs["projmatrix_raw"] = P

        # 断言：如果这里还是没有，直接报我们自己的错
        if "projmatrix_raw" not in kwargs:
            raise RuntimeError("patch failed: projmatrix_raw not injected")

        return orig_forward(viewpoint_camera, bg_color, **kwargs)

    renderer.forward = types.MethodType(forward_patched, renderer)


# ----------------------------
# threestudio-3dgs builders (same style as v29)
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
# Qwen Edit (DiffSynth) loader
# ----------------------------

def load_qwen_pipe(diffsynth_repo_root: str, models_cache_root: str, device: str, torch_dtype: str = "bfloat16"):
    import sys
    if diffsynth_repo_root and os.path.isdir(diffsynth_repo_root) and diffsynth_repo_root not in sys.path:
        sys.path.insert(0, diffsynth_repo_root)

    # Force local caches
    os.environ["MODELSCOPE_CACHE"] = models_cache_root
    os.environ["MODELSCOPE_HOME"] = models_cache_root
    os.environ["MS_CACHE_HOME"] = models_cache_root
    os.environ["HF_HOME"] = models_cache_root
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_cache_root, "transformers")
    os.environ["DIFFSYNTH_MODEL_CACHE"] = models_cache_root

    import torch
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    def _dtype(s: str):
        if s == "bfloat16":
            return torch.bfloat16
        if s == "float16":
            return torch.float16
        if s == "float32":
            return torch.float32
        if s == "float8_e4m3fn":
            return torch.float8_e4m3fn
        raise ValueError(s)

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": _dtype("float8_e4m3fn"),
        "onload_device": "cpu",
        "preparing_dtype": _dtype("float8_e4m3fn"),
        "preparing_device": "cuda",
        "computation_dtype": _dtype(torch_dtype),
        "computation_device": device,
    }

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=_dtype(torch_dtype),
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ],
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    return pipe


def pil_from_rgb01(rgb_hwc: torch.Tensor):
    from PIL import Image

    # Pillow compatibility: Image.Resampling was introduced in Pillow 9.1.0
    if not hasattr(Image, 'Resampling'):
        class _R:
            LANCZOS = Image.LANCZOS
        Image.Resampling = _R
    x = (rgb_hwc.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/nfs4/qhy/projects/threestudio")
    ap.add_argument("--scene_colmap_txt_root", type=str, required=True)
    ap.add_argument("--combined_ply", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--rgb_root", type=str, default="/nfs4/qhy/projects/threestudio/dataset/car_images/input")
    ap.add_argument("--mask_root", type=str, default="/nfs4/qhy/projects/threestudio/dataset/car_images/car_masks")

    ap.add_argument("--max_items", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--bg", type=str, default="1,1,1")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--znear", type=float, default=0.1)
    ap.add_argument("--zfar", type=float, default=100.0)

    # Qwen edit
    ap.add_argument("--diffsynth_repo_root", type=str, default="/nfs4/qhy/projects/DiffSynth-Studio")
    ap.add_argument("--models_cache_root", type=str, default="/nfs4/qhy/projects/DiffSynth-Studio/models/Qwen")
    ap.add_argument("--prompt", type=str, default="把图像编辑得更清晰、更干净，去除噪点和雾气，增强细节与对比度，主体保持不变。")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--denoising_strength", type=float, default=0.8)
    args = ap.parse_args()

    bg = tuple(float(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg must be r,g,b")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "render").mkdir(parents=True, exist_ok=True)
    (out_dir / "qwen_edit").mkdir(parents=True, exist_ok=True)

    # 1) register + build geometry/renderer
    register_needed_modules(args.repo_root)
    geometry = build_geometry(args.combined_ply, sh_degree=args.sh_degree, device=device)
    patch_pc_api(geometry)
    renderer = build_renderer(geometry=geometry, bg_rgb=bg, device=device)
    patch_renderer_forward_add_projraw(renderer, znear=args.znear, zfar=args.zfar)

    # 2) build datamodule + loader
    import threestudio
    DMCls = threestudio.find("colmap-scene-datamodule")
    dm = DMCls({
        "repo_root": args.repo_root,
        "scene_colmap_txt_root": args.scene_colmap_txt_root,
        "rgb_root": args.rgb_root,
        "mask_root": args.mask_root,
        "use_masks": True,
        "batch_size": 1,
        "num_workers": 0,
        "shuffle": False,
        "val_ratio": 0.0,
    })
    dm.setup("fit")
    loader = dm.train_dataloader()

    # 3) load Qwen pipe once
    pipe = load_qwen_pipe(args.diffsynth_repo_root, args.models_cache_root, device=str(device), torch_dtype="bfloat16")

    from PIL import Image

    # Pillow compatibility: Image.Resampling was introduced in Pillow 9.1.0
    if not hasattr(Image, 'Resampling'):
        class _R:
            LANCZOS = Image.LANCZOS
        Image.Resampling = _R

    # 4) iterate
    n = 0
    for batch in loader:
        if n >= args.max_items:
            break
        # datamodule returns camera possibly as list; handle both
        view_id = batch.get("view_id")
        if isinstance(view_id, torch.Tensor):
            view_id = int(view_id[0].item())
        elif isinstance(view_id, (list, tuple)):
            view_id = int(view_id[0])
        else:
            view_id = int(view_id) if view_id is not None else n

        camera = batch.get("camera")
        if isinstance(camera, list):
            viewpoint = camera[0]
        else:
            viewpoint = camera

        name = None
        meta = batch.get("meta")
        if isinstance(meta, dict):
            name = meta.get("image_name")
            if isinstance(name, (list, tuple)):
                name = name[0]
        if name is None:
            name = f"view_{view_id}"
        base = Path(str(name)).stem

        # Render
        with torch.no_grad():
            render_pkg = renderer.forward(viewpoint, renderer.background_tensor)
            rgb_chw = render_pkg["render"]
            rgb = rgb_chw.permute(1, 2, 0).detach().clamp(0, 1)

        render_pil = pil_from_rgb01(rgb)
        render_path = out_dir / "render" / f"{view_id:06d}_{base}.png"
        render_pil.save(render_path)

        # Qwen edit polish (use render as edit_image)
        # Qwen-Image(-Edit) prefers H/W being multiples of 16. If not, it will print:
        #   "height % 16 != 0. We round it up ..."
        # To make the pipeline quiet AND deterministic, we explicitly resize to the rounded-up size
        # before Qwen Edit, then resize the output back to the original render size.
        orig_w, orig_h = render_pil.size
        qwen_w = ((orig_w + 15) // 16) * 16
        qwen_h = ((orig_h + 15) // 16) * 16

        if (qwen_w, qwen_h) != (orig_w, orig_h):
            # LANCZOS gives the best visual quality for resampling.
            render_for_qwen = render_pil.resize((qwen_w, qwen_h), resample=Image.Resampling.LANCZOS)
        else:
            render_for_qwen = render_pil

        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            edit_image=[render_for_qwen],
            num_inference_steps=int(args.num_inference_steps),
            #guidance_scale=float(args.guidance_scale),
            denoising_strength=float(args.denoising_strength),
            height=int(qwen_h),
            width=int(qwen_w),
            #return_dict=True,
        )

        # DiffSynth may return:
        #   - a PipelineOutput with `.images`
        #   - a list/tuple of PIL images
        #   - a single PIL.Image.Image (some versions ignore return_dict)
        #   - a dict with key 'images'
        if hasattr(out, "images"):
            qwen_pil = out.images[0]
        elif isinstance(out, dict) and "images" in out:
            qwen_pil = out["images"][0]
        elif isinstance(out, (list, tuple)):
            qwen_pil = out[0]
        else:
            qwen_pil = out
        if qwen_pil.size != (orig_w, orig_h):
            qwen_pil = qwen_pil.resize((orig_w, orig_h), resample=Image.Resampling.LANCZOS)

        qwen_path = out_dir / "qwen_edit" / f"{view_id:06d}_{base}.png"
        qwen_pil.save(qwen_path)

        print(f"[{n+1}] view_id={view_id} saved: {render_path.name} + {qwen_path.name}")
        n += 1

    print("DONE")
    print(f"Rendered: {out_dir/'render'}")
    print(f"QwenEdit: {out_dir/'qwen_edit'}")


if __name__ == "__main__":
    main()
