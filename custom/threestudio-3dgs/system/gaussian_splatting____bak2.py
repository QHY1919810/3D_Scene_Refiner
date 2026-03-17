import math
from dataclasses import dataclass, field

import os
from pathlib import Path

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud, Camera

# Supervised losses (hard observation anchor)
try:
    from ..utils.sup_losses import compute_sup_losses
except Exception:
    compute_sup_losses = None


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def _camera_to_device(cam: Camera, device: torch.device) -> Camera:
    """Move Camera NamedTuple tensors to device (image_width/height are ints)."""
    return Camera(
        cam.FoVx.to(device),
        cam.FoVy.to(device),
        cam.camera_center.to(device),
        int(cam.image_width),
        int(cam.image_height),
        cam.world_view_transform.to(device),
        cam.full_proj_transform.to(device),
    )


def _ensure_pillow_resampling():
    from PIL import Image

    # Pillow compatibility: Image.Resampling introduced in Pillow 9.1.0
    if not hasattr(Image, "Resampling"):

        class _R:
            LANCZOS = Image.LANCZOS

        Image.Resampling = _R
    return Image


def _pil_from_rgb01(rgb_hwc: torch.Tensor):
    """HWC float 0..1 -> PIL.Image"""
    from PIL import Image

    _ensure_pillow_resampling()
    x = (rgb_hwc.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def _tensor_from_pil(pil_img, device: torch.device, dtype=torch.float32):
    """PIL.Image -> HWC float 0..1 tensor"""
    import numpy as _np

    x = _np.array(pil_img).astype("float32") / 255.0
    if x.ndim == 2:
        x = _np.stack([x, x, x], axis=-1)
    t = torch.from_numpy(x).to(device=device, dtype=dtype)
    return t


class _QwenEditWrapper:
    """Minimal Qwen-Image-Edit-2511 wrapper aligned with your qwenmatch script."""

    def __init__(self, diffsynth_repo_root: str, models_cache_root: str, device: str, torch_dtype: str = "bfloat16"):
        import os, sys

        if diffsynth_repo_root and os.path.isdir(diffsynth_repo_root) and diffsynth_repo_root not in sys.path:
            sys.path.insert(0, diffsynth_repo_root)

        # Force local caches (works regardless of cwd)
        if models_cache_root:
            os.environ["MODELSCOPE_CACHE"] = models_cache_root
            os.environ["MODELSCOPE_HOME"] = models_cache_root
            os.environ["MS_CACHE_HOME"] = models_cache_root
            os.environ["HF_HOME"] = models_cache_root
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_cache_root, "transformers")
            os.environ["DIFFSYNTH_MODEL_CACHE"] = models_cache_root

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

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=_dtype(torch_dtype),
            device=device,
            model_configs=[
                ModelConfig(
                    model_id="Qwen/Qwen-Image-Edit-2511",
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
            ],
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        )

    @torch.no_grad()
    def polish(
        self,
        edit_pil,
        prompt: str,
        negative_prompt: str = "",
        *,
        seed: int = 1,
        num_inference_steps: int = 40,
        denoising_strength: Optional[float] = None,
        edit_image_auto_resize: bool = False,
        zero_cond_t: bool = True,
    ):
        Image = _ensure_pillow_resampling()

        orig_w, orig_h = edit_pil.size
        qwen_w = ((orig_w + 15) // 16) * 16
        qwen_h = ((orig_h + 15) // 16) * 16

        if (qwen_w, qwen_h) != (orig_w, orig_h):
            edit_for_qwen = edit_pil.resize((qwen_w, qwen_h), resample=Image.Resampling.LANCZOS)
        else:
            edit_for_qwen = edit_pil

        kwargs = dict(
            prompt=str(prompt),
            edit_image=[edit_for_qwen],
            seed=int(seed),
            num_inference_steps=int(num_inference_steps),
            height=int(qwen_h),
            width=int(qwen_w),
            edit_image_auto_resize=bool(edit_image_auto_resize),
            zero_cond_t=bool(zero_cond_t),
        )
        if negative_prompt:
            kwargs["negative_prompt"] = str(negative_prompt)
        if denoising_strength is not None:
            kwargs["denoising_strength"] = float(denoising_strength)

        out = self.pipe(**kwargs)

        if hasattr(out, "images"):
            pil = out.images[0]
        elif isinstance(out, dict) and "images" in out:
            pil = out["images"][0]
        elif isinstance(out, (list, tuple)):
            pil = out[0]
        else:
            pil = out

        if pil.size != (orig_w, orig_h):
            pil = pil.resize((orig_w, orig_h), resample=Image.Resampling.LANCZOS)
        return pil


@threestudio.register("gaussian-splatting-system-bak2")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

        # Switches
        enable_sds: bool = True
        enable_sup: bool = False
        enable_rewrite: bool = False

        # Debug image dumping (render + qwen_edit) during training
        save_debug_images: bool = True
        save_debug_interval: int = 200           # save every N global steps
        save_debug_include_step: bool = True     # filename includes global_step
        save_debug_dir: str = ""                 # empty -> use trainer.log_dir/debug_images; or set absolute/relative path


        # Rewrite cache
        rewrite_update_interval: int = 100
        rewrite_edit_source: str = "render"   # "render" or "gt"
        rewrite_loss_type: str = "l2"         # "l1" or "l2"
        rewrite_masked: bool = True

        # Qwen config (defaults match your qwenmatch script)
        qwen: dict = field(default_factory=lambda: {
            "diffsynth_repo_root": "/nfs4/qhy/projects/DiffSynth-Studio",
            "models_cache_root": "/nfs4/qhy/projects/DiffSynth-Studio/models/Qwen",
            "torch_dtype": "bfloat16",
            "seed": 1,
            "num_inference_steps": 40,
            #"denoising_strength": 0.8,
            "zero_cond_t": True,
            "edit_image_auto_resize": True,
            "negative_prompt": "",
        })

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.automatic_optimization = False

        # Prompt processor
        if hasattr(self.cfg, "prompt_processor_type") and self.cfg.prompt_processor_type is not None:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.prompt_utils = self.prompt_processor()
        else:
            self.prompt_processor = None
            self.prompt_utils = None

        # Guidance (legacy SDS)
        if getattr(self.cfg, "enable_sds", True):
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        else:
            self.guidance = None

        # Qwen rewrite
        self._qwen_wrapper = None
        self._rewrite_cache: Dict[str, Tuple[int, Any]] = {}

        # Ensure geometry provides get_features if renderer.forward needs it
        self._ensure_geometry_feature_getter()

        # Patch renderer.forward to always provide projmatrix_raw (some forks require it)
        self._patch_renderer_forward_add_projraw()

    def _ensure_geometry_feature_getter(self):
        pc = self.geometry
        if hasattr(pc, "get_features"):
            return

        def _get_features(self):
            if hasattr(self, "_features_dc") and hasattr(self, "_features_rest"):
                return torch.cat([self._features_dc, self._features_rest], dim=1)
            if hasattr(self, "_features"):
                return self._features
            if hasattr(self, "features"):
                return self.features
            raise AttributeError("Geometry has no features tensors for get_features.")

        try:
            setattr(pc.__class__, "get_features", property(_get_features))
        except Exception:
            pass

    def _patch_renderer_forward_add_projraw(self):
        import types
        import torch as _torch
        import math as _math

        if not hasattr(self.renderer, "forward"):
            return
        if getattr(self.renderer, "_projraw_patched", False):
            return

        orig_forward = self.renderer.forward
        znear = float(getattr(self.cfg, "znear", 0.1))
        zfar = float(getattr(self.cfg, "zfar", 100.0))

        def _proj_raw_from_fov(fovx, fovy, device):
            tanfovx = _math.tan(float(fovx) * 0.5)
            tanfovy = _math.tan(float(fovy) * 0.5)
            P = _torch.zeros((4, 4), device=device, dtype=_torch.float32)
            P[0, 0] = 1.0 / (tanfovx + 1e-9)
            P[1, 1] = 1.0 / (tanfovy + 1e-9)
            P[2, 2] = float(zfar) / (float(zfar) - float(znear) + 1e-9)
            P[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear) + 1e-9)
            P[3, 2] = 1.0
            return P

        def forward_patched(this, viewpoint_camera, bg_color, **kwargs):
            dev = bg_color.device if isinstance(bg_color, _torch.Tensor) else _torch.device("cuda")
            P = _proj_raw_from_fov(viewpoint_camera.FoVx, viewpoint_camera.FoVy, dev).transpose(0, 1)
            kwargs["projmatrix_raw"] = P
            return orig_forward(viewpoint_camera, bg_color, **kwargs)

        self.renderer.forward = types.MethodType(forward_patched, self.renderer)
        setattr(self.renderer, "_projraw_patched", True)

    def _get_prompt_text(self) -> str:
        if self.prompt_utils is not None and hasattr(self.prompt_utils, "prompt"):
            p = self.prompt_utils.prompt
            if isinstance(p, (list, tuple)):
                return str(p[0])
            return str(p)
        try:
            return str(self.cfg.prompt_processor.prompt)
        except Exception:
            return ""

    def _maybe_init_qwen(self):
        if self._qwen_wrapper is not None:
            return
        qcfg = dict(getattr(self.cfg, "qwen", {}) or {})
        self._qwen_wrapper = _QwenEditWrapper(
            diffsynth_repo_root=qcfg.get("diffsynth_repo_root"),
            models_cache_root=qcfg.get("models_cache_root"),
            device=str(self.device),
            torch_dtype=qcfg.get("torch_dtype", "bfloat16"),
        )

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)

        cams = batch.get("camera", None)
        if cams is not None:
            cams = _as_list(cams)
            comp_rgbs, comp_masks = [], []
            visibility_filters, radii_list, viewspace_points = [], [], []

            for cam in cams:
                cam_dev = _camera_to_device(cam, self.device)
                render_pkg = self.renderer.forward(cam_dev, self.renderer.background_tensor)
                rgb_chw = render_pkg["render"]
                comp_rgbs.append(rgb_chw.permute(1, 2, 0).contiguous())
                if "mask" in render_pkg:
                    comp_masks.append(render_pkg["mask"].permute(1, 2, 0).contiguous())
                if "visibility_filter" in render_pkg:
                    visibility_filters.append(render_pkg["visibility_filter"])
                if "radii" in render_pkg:
                    radii_list.append(render_pkg["radii"])
                if "viewspace_points" in render_pkg:
                    viewspace_points.append(render_pkg["viewspace_points"])

            out = {"comp_rgb": torch.stack(comp_rgbs, dim=0)}
            if len(comp_masks) == len(comp_rgbs):
                out["comp_mask"] = torch.stack(comp_masks, dim=0)
            if visibility_filters:
                out["visibility_filter"] = visibility_filters
            if radii_list:
                out["radii"] = radii_list
            if viewspace_points:
                out["viewspace_points"] = viewspace_points
            return out

        return self.renderer.batch_forward(batch)

    def _rewrite_cache_key(self, view_id: Any, prompt: str) -> str:
        if isinstance(view_id, torch.Tensor):
            view_id = int(view_id.flatten()[0].item())
        elif isinstance(view_id, (list, tuple)) and len(view_id) > 0:
            view_id = int(view_id[0])
        elif view_id is None:
            view_id = -1
        return f"{int(view_id)}::{prompt}"

    def _debug_base_dir(self) -> Path:
        """
        Return base directory for dumping debug images.
        Priority:
          1) cfg.save_debug_dir (absolute or relative to trainer.log_dir/default_root_dir)
          2) trainer.log_dir/debug_images
          3) default_root_dir/debug_images
          4) ./outputs/debug_images
        """
        # 1) user-specified dir
        d = str(getattr(self.cfg, "save_debug_dir", "") or "").strip()
        if d:
            p = Path(d)
            if p.is_absolute():
                return p
            base = getattr(self.trainer, "log_dir", None) or getattr(self.trainer, "default_root_dir", None) or "outputs"
            return Path(base) / d

        # 2/3/4) fallbacks
        base = getattr(self.trainer, "log_dir", None) or getattr(self.trainer, "default_root_dir", None) or "outputs"
        return Path(base) / "debug_images"

    def _maybe_dump_render_and_qwen(self, render_hwc: torch.Tensor, qwen_hwc: torch.Tensor, batch: Dict[str, Any]) -> None:
        if not bool(getattr(self.cfg, "save_debug_images", False)):
            return
        interval = int(getattr(self.cfg, "save_debug_interval", 200))
        step = int(self.global_step)
        if interval > 0 and (step % interval) != 0:
            return

        # avoid duplicate dumps under DDP
        if hasattr(self, "global_rank") and int(getattr(self, "global_rank", 0)) != 0:
            return

        # resolve view_id
        view_id = batch.get("view_id", None)
        if isinstance(view_id, torch.Tensor):
            view_id = int(view_id.flatten()[0].item())
        elif isinstance(view_id, (list, tuple)) and len(view_id) > 0:
            view_id = int(view_id[0])
        elif view_id is None:
            view_id = -1

        # resolve base name (same logic as your qwenmatch sanity script)
        name = None
        meta = batch.get("meta", None)
        if isinstance(meta, dict):
            name = meta.get("image_name", None)
            if isinstance(name, (list, tuple)):
                name = name[0] if len(name) > 0 else None
        if name is None:
            name = f"view_{view_id}"
        base = Path(str(name)).stem

        base_dir = self._debug_base_dir()
        render_dir = base_dir / "render"
        qwen_dir = base_dir / "qwen_edit"
        render_dir.mkdir(parents=True, exist_ok=True)
        qwen_dir.mkdir(parents=True, exist_ok=True)

        prefix = ""
        if bool(getattr(self.cfg, "save_debug_include_step", True)):
            prefix = f"{step:07d}_"

        render_pil = _pil_from_rgb01(render_hwc.detach())
        qwen_pil = _pil_from_rgb01(qwen_hwc.detach())

        render_path = render_dir / f"{prefix}{view_id:06d}_{base}.png"
        qwen_path = qwen_dir / f"{prefix}{view_id:06d}_{base}.png"
        render_pil.save(render_path)
        qwen_pil.save(qwen_path)
    @torch.no_grad()
    def _get_qwen_target_hwc(self, edit_source_rgb_hwc: torch.Tensor, prompt: str, view_id: Any) -> torch.Tensor:
        self._maybe_init_qwen()
        qcfg = dict(getattr(self.cfg, "qwen", {}) or {})

        key = self._rewrite_cache_key(view_id, prompt)
        step = int(self.global_step)
        upd = int(getattr(self.cfg, "rewrite_update_interval", 100))

        if key in self._rewrite_cache:
            last_step, tgt = self._rewrite_cache[key]
            if (step - int(last_step)) < upd:
                return tgt

        edit_pil = _pil_from_rgb01(edit_source_rgb_hwc)
        print(str(qcfg.get("prompt", "")))
        print(str(qcfg.get("negative_prompt", "")))

        tgt_pil = self._qwen_wrapper.polish(
            edit_pil=edit_pil,
            prompt=str(qcfg.get("prompt", "")),
            negative_prompt=str(qcfg.get("negative_prompt", "")),
            seed=int(qcfg.get("seed", 1)),
            num_inference_steps=int(qcfg.get("num_inference_steps", 40)),
            denoising_strength=float(qcfg.get("denoising_strength", 0.8)) if "denoising_strength" in qcfg else None,
            edit_image_auto_resize=bool(qcfg.get("edit_image_auto_resize", True)),
            zero_cond_t=bool(qcfg.get("zero_cond_t", True)),
        )
        tgt = _tensor_from_pil(tgt_pil, device=self.device, dtype=torch.float32)
        self._rewrite_cache[key] = (step, tgt)
        return tgt

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)

        loss_total = torch.tensor(0.0, device=self.device)

        # SDS (optional)
        if getattr(self.cfg, "enable_sds", True) and self.guidance is not None:
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False)
            loss_sds = torch.tensor(0.0, device=self.device)
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds = loss_sds + value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            loss_total = loss_total + loss_sds

        # Supervised (hard)
        if getattr(self.cfg, "enable_sup", False):
            if compute_sup_losses is None:
                raise ImportError("compute_sup_losses not available. Ensure utils/sup_losses.py exists.")
            if "rgb" not in batch:
                raise KeyError("enable_sup=True requires batch['rgb'] from datamodule.")

            sup = compute_sup_losses(
                outputs=out,
                batch=batch,
                lambda_rgb=float(self.cfg.loss.get("lambda_rgb", 1.0)),
                lambda_silhouette=float(self.cfg.loss.get("lambda_silhouette", 0.0)),
                rgb_loss_type=str(getattr(self.cfg, "sup_rgb_loss_type", "l1")),
                mask_dilate=int(getattr(self.cfg, "sup_mask_dilate", 0)),
            )
            self.log("train/loss_rgb", sup["loss_rgb"])
            self.log("train/loss_silhouette", sup["loss_silhouette"])
            loss_total = loss_total + sup["loss_sup"]

        # Qwen rewrite (soft)
        if getattr(self.cfg, "enable_rewrite", False):
            #prompt = self.cfg.qwen.prompt
            view_id = batch.get("view_id", None)

            rgb_pred = out["comp_rgb"]
            if rgb_pred.shape[0] != 1:
                raise ValueError("enable_rewrite currently expects batch_size=1.")

            src = str(getattr(self.cfg, "rewrite_edit_source", "render")).lower()
            if src == "gt":
                if "rgb" not in batch:
                    raise KeyError("rewrite_edit_source='gt' requires batch['rgb'].")
                edit_src = batch["rgb"][0].to(device=self.device, dtype=torch.float32)
            else:
                edit_src = rgb_pred[0].to(dtype=torch.float32)

            tgt = self._get_qwen_target_hwc(edit_src, prompt="", view_id=view_id)
            # Debug dump: render + qwen edit (no impact on training)
            self._maybe_dump_render_and_qwen(rgb_pred[0], tgt, batch)


            # loss
            loss_type = str(getattr(self.cfg, "rewrite_loss_type", "l2")).lower()
            w = None
            if getattr(self.cfg, "rewrite_masked", True) and ("mask" in batch):
                mask = batch["mask"][0].to(device=self.device, dtype=torch.float32)
                if mask.ndim == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]
                w = mask.expand_as(rgb_pred[0])

            pred = rgb_pred[0].to(dtype=torch.float32)
            if w is not None:
                denom = w.sum() + 1e-8
                if loss_type == "l1":
                    loss_rw = ((pred - tgt).abs() * w).sum() / denom
                else:
                    loss_rw = (((pred - tgt) ** 2) * w).sum() / denom
            else:
                if loss_type == "l1":
                    loss_rw = (pred - tgt).abs().mean()
                else:
                    loss_rw = F.mse_loss(pred, tgt)

            lam = float(self.cfg.loss.get("lambda_rewrite", 0.1))
            self.log("train/loss_rewrite", loss_rw)
            loss_total = loss_total + lam * loss_rw

        # Regularizers
        if self.cfg.loss.get("lambda_position", 0.0) > 0.0:
            loss_position = self.geometry.get_xyz.norm(dim=-1).mean()
            self.log("train/loss_position", loss_position)
            loss_total = loss_total + self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss.get("lambda_opacity", 0.0) > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (scaling.detach().unsqueeze(-1) * self.geometry.get_opacity).sum()
            self.log("train/loss_opacity", loss_opacity)
            loss_total = loss_total + self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss.get("lambda_scales", 0.0) > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log("train/scales", scale_sum)
            loss_total = loss_total + self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss.get("lambda_tv_loss", 0.0) > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
            self.log("train/loss_tv", loss_tv)
            loss_total = loss_total + loss_tv

        if out.__contains__("comp_depth") and self.cfg.loss.get("lambda_depth_tv_loss", 0.0) > 0.0:
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2)) + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log("train/loss_depth_tv", loss_depth_tv)
            loss_total = loss_total + loss_depth_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_total.backward(retain_graph=True)

        iteration = self.global_step
        if "visibility_filter" in out and "radii" in out and "viewspace_points" in out:
            self.geometry.update_states(iteration, out["visibility_filter"], out["radii"], out["viewspace_points"])

        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_total}

    def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(points=np.zeros((num_pts, 3)), colors=np.zeros((num_pts, 3)), normals=np.zeros((num_pts, 3)))
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)
