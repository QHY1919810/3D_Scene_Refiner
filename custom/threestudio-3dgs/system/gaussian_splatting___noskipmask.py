import math
from dataclasses import dataclass, field

import os
from pathlib import Path

import numpy as np
import threestudio
import torch
import torch.nn.functional as F

def _resize_mask_to_hw(mask_hwc: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Resize a HxWx1 (or HxW) mask to (H,W) using nearest."""
    if mask_hwc.ndim == 2:
        mask_hwc = mask_hwc[..., None]
    mask_hwc = mask_hwc[..., :1]
    h0, w0 = int(mask_hwc.shape[0]), int(mask_hwc.shape[1])
    if h0 == H and w0 == W:
        return mask_hwc
    m = mask_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,1,h,w
    m = F.interpolate(m, size=(H, W), mode="nearest")
    return m.squeeze(0).permute(1, 2, 0).contiguous()

from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud, Camera
import matplotlib.pyplot as plt

# Supervised losses (hard observation anchor)
try:
    from ..utils.sup_losses import compute_sup_losses
except Exception:
    compute_sup_losses = None


# ---------------------------
# High-frequency (HF) loss (masked)
# ---------------------------
def _hf_response_map(
    rgb_bhwc: torch.Tensor,
    mode: str = "laplacian",
    grayscale: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a per-pixel high-frequency response map.
    Args:
      rgb_bhwc: [B,H,W,3] float in [0,1]
    Returns:
      hf: [B,H,W,1] float
    """
    assert rgb_bhwc.ndim == 4 and rgb_bhwc.shape[-1] == 3, f"expected [B,H,W,3], got {rgb_bhwc.shape}"
    x = rgb_bhwc.permute(0, 3, 1, 2)  # B,3,H,W

    if grayscale:
        # luminance
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        x = 0.2989 * r + 0.5870 * g + 0.1140 * b  # B,1,H,W

    mode = str(mode).lower()
    if mode == "sobel":
        # Sobel gradients
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

        if x.shape[1] == 1:
            gx = F.conv2d(x, kx, padding=1)
            gy = F.conv2d(x, ky, padding=1)
            mag = torch.sqrt(gx * gx + gy * gy + eps)
            hf = mag
        else:
            # apply per-channel then sum magnitudes
            mags = []
            for c in range(x.shape[1]):
                xc = x[:, c:c+1]
                gx = F.conv2d(xc, kx, padding=1)
                gy = F.conv2d(xc, ky, padding=1)
                mags.append(torch.sqrt(gx * gx + gy * gy + eps))
            hf = torch.sum(torch.cat(mags, dim=1), dim=1, keepdim=True)
    else:
        # Laplacian (4-neighbor), stable and cheap
        k = torch.tensor([[0.0,  1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0,  1.0, 0.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

        if x.shape[1] == 1:
            hf = F.conv2d(x, k, padding=1).abs()
        else:
            # per-channel then sum abs
            outs = []
            for c in range(x.shape[1]):
                outs.append(F.conv2d(x[:, c:c+1], k, padding=1).abs())
            hf = torch.sum(torch.cat(outs, dim=1), dim=1, keepdim=True)

    return hf.permute(0, 2, 3, 1).contiguous()  # B,H,W,1


def masked_high_frequency_loss(
    pred_rgb_bhwc: torch.Tensor,
    gt_rgb_bhwc: torch.Tensor,
    mask_bhw1: Optional[torch.Tensor] = None,
    *,
    mode: str = "laplacian",
    loss_type: str = "l1",
    grayscale: bool = True,
    mask_dilate: int = 0,
) -> torch.Tensor:
    """
    High-frequency loss inside mask.
    - pred/gt: [B,H,W,3]
    - mask: [B,H,W,1] or [B,H,W] (1=keep). If None -> full image.
    """
    hf_pred = _hf_response_map(pred_rgb_bhwc, mode=mode, grayscale=grayscale)
    hf_gt = _hf_response_map(gt_rgb_bhwc.detach(), mode=mode, grayscale=grayscale)

    diff = (hf_pred - hf_gt)
    loss_type = str(loss_type).lower()
    if loss_type == "l2" or loss_type == "mse":
        diff = diff * diff
    else:
        diff = diff.abs()

    if mask_bhw1 is None:
        return diff.mean()

    if mask_bhw1.ndim == 3:
        mask_bhw1 = mask_bhw1.unsqueeze(-1)
    mask_bhw1 = mask_bhw1.to(device=diff.device, dtype=diff.dtype)[..., :1]

    if mask_dilate > 0:
        m = mask_bhw1.permute(0, 3, 1, 2)  # B,1,H,W
        m = F.max_pool2d(m, kernel_size=mask_dilate * 2 + 1, stride=1, padding=mask_dilate)
        mask_bhw1 = m.permute(0, 2, 3, 1)

    # match mask size to diff if needed
    if mask_bhw1.shape[1] != diff.shape[1] or mask_bhw1.shape[2] != diff.shape[2]:
        m = mask_bhw1.permute(0, 3, 1, 2)
        m = F.interpolate(m, size=(diff.shape[1], diff.shape[2]), mode="nearest")
        mask_bhw1 = m.permute(0, 2, 3, 1).contiguous()
    w = mask_bhw1.expand_as(diff)
    return (diff * w).sum() / (w.sum() + 1e-8)




def masked_img_loss_hwc(
    pred_hwc: torch.Tensor,
    tgt_hwc: torch.Tensor,
    mask_hw1: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    if mask_hw1.ndim == 2:
        mask_hw1 = mask_hw1[..., None]
    mask_hw1 = mask_hw1[..., :1].to(device=pred_hwc.device, dtype=pred_hwc.dtype)
    w = mask_hw1.expand_as(pred_hwc)
    denom = w.sum() + 1e-8

    if loss_type == "l1":
        return ((pred_hwc - tgt_hwc).abs() * w).sum() / denom
    else:
        return (((pred_hwc - tgt_hwc) ** 2) * w).sum() / denom

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


@threestudio.register("gaussian-splatting-system")
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

        # Periodic PLY snapshots (training target is the point cloud)
        # 0 disables periodic saving.
        save_ply_every_n_steps: int = 0
        # If true, always overwrite the same filename "point_cloud.ply".
        # If false, write "point_cloud_stepXXXXXXX.ply" snapshots.
        save_ply_overwrite: bool = True
        # If non-empty, save PLYs under this directory (absolute or relative).
        # If empty, use the experiment save directory via self.get_save_path(...).
        save_ply_dir: str = ""

        # Loss curve visualization (overwrite a single PNG each update)
        loss_plot_enable: bool = True
        loss_plot_every_n_steps: int = 1
        loss_plot_max_points: int = 500
        # If empty, defaults to <save_debug_dir>/loss_curve.png (or experiment save dir if save_debug_dir empty)
        loss_plot_path: str = ""


        # Rewrite cache
        rewrite_update_interval: int = 100
        rewrite_edit_source: str = "render"   # "render" or "gt"
        rewrite_loss_type: str = "l2"         # "l1" or "l2"
        rewrite_masked: bool = True

        # HF loss (masked high-frequency anchor inside mask)
        hf_mode: str = "laplacian"     # "laplacian" or "sobel"
        hf_loss_type: str = "l1"       # "l1" or "l2"
        hf_grayscale: bool = True
        hf_mask_dilate: int = 0

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

        # Fixed teacher mode (optional): if enabled, read teacher PNGs from disk by view_id.
        use_fixed_rewrite_targets: bool = False
        fixed_rewrite_target_dir: str = ""
        save_fixed_rewrite_targets: bool = False

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
    def _fixed_target_path(self, view_id: Any) -> str:
        base = str(getattr(self.cfg, "fixed_rewrite_target_dir", "") or "")
        if base == "":
            return ""
        if isinstance(view_id, torch.Tensor):
            view_id = int(view_id.flatten()[0].item())
        elif isinstance(view_id, (list, tuple)) and len(view_id) > 0:
            view_id = int(view_id[0])
        elif view_id is None:
            view_id = -1
        return os.path.join(base, f"view_{int(view_id)}.png")

    @torch.no_grad()
    def _load_fixed_target_hwc(self, view_id: Any) -> Optional[torch.Tensor]:
        path = self._fixed_target_path(view_id)
        if path == "" or (not os.path.isfile(path)):
            return None
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        return _tensor_from_pil(pil, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _save_fixed_target_hwc(self, tgt_hwc: torch.Tensor, view_id: Any) -> None:
        path = self._fixed_target_path(view_id)
        if path == "":
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil = _pil_from_rgb01(tgt_hwc.detach())
        pil.save(path)

    @torch.no_grad()
    def _get_qwen_target_hwc(self, edit_source_rgb_hwc: torch.Tensor, prompt: str, view_id: Any) -> torch.Tensor:
        if bool(getattr(self.cfg, "use_fixed_rewrite_targets", False)):
            tgt_disk = self._load_fixed_target_hwc(view_id)
            if tgt_disk is not None:
                return tgt_disk

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
        if bool(getattr(self.cfg, "save_fixed_rewrite_targets", False)):
            self._save_fixed_target_hwc(tgt, view_id)
        return tgt

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)

        if getattr(self.cfg, "enable_sup", False):
            raise RuntimeError("This point-routed version removes sup loss. Set enable_sup=False.")
        if getattr(self.cfg, "enable_sds", False) and self.guidance is not None:
            raise RuntimeError("This point-routed version does not support shared SDS loss. Disable enable_sds.")
        if not getattr(self.cfg, "enable_rewrite", False):
            raise RuntimeError("This point-routed version requires enable_rewrite=True.")
        if "mask" not in batch or batch["mask"] is None:
            raise RuntimeError("This point-routed version requires batch['mask'].")
        if "rgb" not in batch:
            raise KeyError("HF loss requires batch['rgb'] from datamodule.")

        # Strict point-routed training should not mix shared/global losses.
        shared_loss_names = [
            "lambda_position",
            "lambda_opacity",
            "lambda_scales",
            "lambda_tv_loss",
            "lambda_depth_tv_loss",
            "lambda_rgb",
            "lambda_silhouette",
        ]
        enabled_shared = [n for n in shared_loss_names if float(self.cfg.loss.get(n, 0.0)) > 0.0]
        if enabled_shared:
            raise RuntimeError(
                "This point-routed version requires shared/global losses to be disabled. "
                f"Please set these to 0: {enabled_shared}"
            )

        rgb_pred = out["comp_rgb"]   # [B,H,W,3]
        if rgb_pred.shape[0] != 1:
            raise ValueError("point-routed rewrite currently expects batch_size=1.")

        pred = rgb_pred[0].to(dtype=torch.float32)
        H, W = int(pred.shape[0]), int(pred.shape[1])

        if not torch.any(self.geometry.car_point_mask):
            raise RuntimeError("No car points found in geometry._point_tag.")
        if not torch.any(self.geometry.bg_point_mask):
            raise RuntimeError("No bg points found in geometry._point_tag.")

        # ------------------------------------------------------------
        # current view target for rewrite
        # ------------------------------------------------------------
        view_id = batch.get("view_id", None)
        src = str(getattr(self.cfg, "rewrite_edit_source", "render")).lower()

        if src == "gt":
            edit_src = batch["rgb"][0].to(device=self.device, dtype=torch.float32)
        else:
            edit_src = pred

        tgt = self._get_qwen_target_hwc(edit_src, prompt="", view_id=view_id)
        self._maybe_dump_render_and_qwen(pred, tgt, batch)

        # ------------------------------------------------------------
        # build mask_in / mask_out at render resolution
        # ------------------------------------------------------------
        mask = batch["mask"][0].to(device=self.device, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = mask[..., :1]

        mask_in = _resize_mask_to_hw(mask, H, W)
        mask_in = (mask_in > 0.5).to(dtype=pred.dtype)
        mask_out = 1.0 - mask_in

        # ------------------------------------------------------------
        # rewrite losses
        # ------------------------------------------------------------
        rw_loss_type = str(getattr(self.cfg, "rewrite_loss_type", "l2")).lower()
        loss_rw_in = masked_img_loss_hwc(pred_hwc=pred, tgt_hwc=tgt, mask_hw1=mask_in, loss_type=rw_loss_type)
        loss_rw_out = masked_img_loss_hwc(pred_hwc=pred, tgt_hwc=tgt, mask_hw1=mask_out, loss_type=rw_loss_type)

        # ------------------------------------------------------------
        # HF loss only inside mask
        # ------------------------------------------------------------
        lam_hf = float(self.cfg.loss.get("lambda_hf", 0.0))
        if lam_hf > 0.0:
            loss_hf_in = masked_high_frequency_loss(
                pred_rgb_bhwc=rgb_pred.to(dtype=torch.float32),
                gt_rgb_bhwc=batch["rgb"].to(device=self.device, dtype=torch.float32),
                mask_bhw1=batch["mask"].to(device=self.device, dtype=torch.float32),
                mode=str(getattr(self.cfg, "hf_mode", "laplacian")),
                loss_type=str(getattr(self.cfg, "hf_loss_type", "l1")),
                grayscale=bool(getattr(self.cfg, "hf_grayscale", True)),
                mask_dilate=int(getattr(self.cfg, "hf_mask_dilate", 0)),
            )
        else:
            loss_hf_in = pred.sum() * 0.0

        lam_rw = float(self.cfg.loss.get("lambda_rewrite", 0.1))

        self.log("train/loss_rewrite_in", loss_rw_in)
        self.log("train/loss_rewrite_out", loss_rw_out)
        self.log("train/loss_hf", loss_hf_in)
        self._record_loss_value("loss_rewrite_in", loss_rw_in)
        self._record_loss_value("loss_rewrite_out", loss_rw_out)
        self._record_loss_value("loss_hf", loss_hf_in)

        opt.zero_grad(set_to_none=True)
        routed_names = ["_features_dc", "_features_rest"]

        # ============================================================
        # branch A: car points <- (mask-in rewrite + mask-in HF)
        # ============================================================
        loss_car = lam_rw * loss_rw_in + lam_hf * loss_hf_in
        self.manual_backward(loss_car, retain_graph=True)
        self.geometry.mask_param_grad_inplace(
            keep_mask_1d=self.geometry.car_point_mask,
            names=routed_names,
        )
        car_grads = self.geometry.clone_current_grads(names=routed_names)

        opt.zero_grad(set_to_none=True)

        # ============================================================
        # branch B: bg points <- mask-out rewrite
        # ============================================================
        loss_bg = lam_rw * loss_rw_out
        self.manual_backward(loss_bg, retain_graph=True)
        self.geometry.mask_param_grad_inplace(
            keep_mask_1d=self.geometry.bg_point_mask,
            names=routed_names,
        )
        bg_grads = self.geometry.clone_current_grads(names=routed_names)

        opt.zero_grad(set_to_none=True)

        # ============================================================
        # merge routed grads
        # ============================================================
        self.geometry.assign_grads_from_dict(car_grads)
        self.geometry.add_grads_from_dict(bg_grads)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        opt.step()
        opt.zero_grad(set_to_none=True)

        iteration = self.global_step
        if "visibility_filter" in out and "radii" in out and "viewspace_points" in out:
            1
            # self.geometry.update_states(iteration, out["visibility_filter"], out["radii"], out["viewspace_points"])

        loss_total = loss_car.detach() + loss_bg.detach()
        self._record_loss_value("loss_total", loss_total)
        return {"loss": loss_total}
    def _record_loss_value(self, name: str, value) -> None:
        """Record a scalar loss value for plotting (rank0 only)."""
        if getattr(self, "global_rank", 0) != 0:
            return
        try:
            v = float(value.detach().item()) if hasattr(value, "detach") else float(value)
        except Exception:
            return
        if not hasattr(self, "_loss_hist"):
            self._loss_hist = {}
        self._loss_hist.setdefault(name, []).append((int(self.global_step), v))

    def _maybe_save_loss_plot(self) -> None:
        """Save loss curve PNG (overwrite) every N steps (rank0 only)."""
        if getattr(self, "global_rank", 0) != 0:
            return
        if not bool(getattr(self.cfg, "loss_plot_enable", True)):
            return
        every = int(getattr(self.cfg, "loss_plot_every_n_steps", 1) or 1)
        step = int(self.global_step)
        if step <= 0 or (step % every) != 0:
            return

        if not hasattr(self, "_loss_hist") or len(self._loss_hist) == 0:
            return

        max_pts = int(getattr(self.cfg, "loss_plot_max_points", 500) or 500)

        save_path = str(getattr(self.cfg, "loss_plot_path", "") or "").strip()
        if not save_path:
            base = str(getattr(self.cfg, "save_debug_dir", "") or "").strip()
            if base:
                os.makedirs(base, exist_ok=True)
                save_path = os.path.join(base, "loss_curve.png")
            else:
                save_path = self.get_save_path("loss_curve.png")

        plt.figure(figsize=(8, 4))
        for k, pts in sorted(self._loss_hist.items()):
            if not pts:
                continue
            pts2 = pts[-max_pts:]
            xs = [p[0] for p in pts2]
            ys = [p[1] for p in pts2]
            plt.plot(xs, ys, label=k)
        plt.xlabel("global_step")
        plt.ylabel("loss")
        plt.title("Training losses (recent)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # Periodic PLY saving (rank 0 only)
        n = int(getattr(self.cfg, "save_ply_every_n_steps", 0) or 0)
        if n <= 0:
            return
        step = int(self.global_step)
        if step <= 0 or (step % n) != 0:
            return
        if getattr(self, "global_rank", 0) != 0:
            return

        overwrite = bool(getattr(self.cfg, "save_ply_overwrite", True))
        save_dir = str(getattr(self.cfg, "save_ply_dir", "") or "").strip()

        fname = "point_cloud.ply" if overwrite else f"point_cloud_step{step:07d}.ply"
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)
        else:
            save_path = self.get_save_path(fname)

        try:
            self.geometry.save_ply(save_path)
            print(f"[PLY] saved: {save_path}")
        except Exception as e:
            print(f"[PLY] save failed at step {step}: {e}")
        self._maybe_save_loss_plot()
        return


    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Minimal validation_step to satisfy Lightning/Threestudio BaseSystem.
        We do NOT run optimization here.
        """
        with torch.no_grad():
            out = self.forward(batch)

            # Optional: log a simple validation RGB error if GT is present.
            if "rgb" in batch:
                pred = out.get("comp_rgb", None)
                if pred is not None:
                    gt = batch["rgb"].to(device=pred.device, dtype=pred.dtype)
                    if "mask" in batch and batch["mask"] is not None:
                        m = batch["mask"].to(device=pred.device, dtype=pred.dtype)
                        if m.ndim == 3:
                            m = m.unsqueeze(-1)
                        diff = (pred - gt).abs()
                        # match mask resolution to pred
                        if m.shape[1] != diff.shape[1] or m.shape[2] != diff.shape[2]:
                            mm = m.permute(0, 3, 1, 2)
                            mm = F.interpolate(mm, size=(diff.shape[1], diff.shape[2]), mode="nearest")
                            m = mm.permute(0, 2, 3, 1).contiguous()
                        w = m.expand_as(diff)
                        loss_rgb = (diff * w).sum() / (w.sum() + 1e-8)
                    else:
                        loss_rgb = (pred - gt).abs().mean()
                    self.log("val/loss_rgb", loss_rgb, prog_bar=True)

            return {"val_out": 0.0}

def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(points=np.zeros((num_pts, 3)), colors=np.zeros((num_pts, 3)), normals=np.zeros((num_pts, 3)))
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)
