import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer


def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    """Build the *raw* projection matrix (no view).

    Matches the v29 working patch behavior (numpy version), but returns torch.
    Returned matrix is NOT transposed.
    """
    tan_half_fovx = math.tan(float(fovx) * 0.5)
    tan_half_fovy = math.tan(float(fovy) * 0.5)
    P = torch.zeros((4, 4), dtype=torch.float32)
    P[0, 0] = 1.0 / (tan_half_fovx + 1e-9)
    P[1, 1] = 1.0 / (tan_half_fovy + 1e-9)
    P[2, 2] = float(zfar) / (float(zfar) - float(znear) + 1e-9)
    P[2, 3] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear) + 1e-9)
    P[3, 2] = 1.0
    return P


@threestudio.register("diff-gaussian-rasterizer")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        # Defaults chosen to match your v29 pipeline / camera builder defaults.
        znear: float = 0.1
        zfar: float = 100.0

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Render the scene.

        Patch vs original:
        - Your `GaussianRasterizationSettings` requires `projmatrix_raw`.
          The original implementation didn't pass it, causing:
            TypeError: ... missing 1 required positional argument: 'projmatrix_raw'
          We compute it (or accept it from kwargs) and pass it explicitly.
        """

        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry
        dev = pc.get_xyz.device

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=dev
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = math.tan(float(viewpoint_camera.FoVx) * 0.5)
        tanfovy = math.tan(float(viewpoint_camera.FoVy) * 0.5)

        # --- REQUIRED by your GaussianRasterizationSettings ---
        projmatrix_raw = kwargs.get("projmatrix_raw", None)
        if projmatrix_raw is None:
            znear = float(kwargs.get("znear", self.cfg.znear))
            zfar = float(kwargs.get("zfar", self.cfg.zfar))
            P = get_projection_matrix(znear, zfar, float(viewpoint_camera.FoVx), float(viewpoint_camera.FoVy))
            projmatrix_raw = P.to(device=dev, dtype=torch.float32).transpose(0, 1)
        else:
            projmatrix_raw = projmatrix_raw.to(device=dev, dtype=torch.float32)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color.to(device=dev, dtype=torch.float32),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(device=dev, dtype=torch.float32),
            projmatrix=viewpoint_camera.full_proj_transform.to(device=dev, dtype=torch.float32),
            projmatrix_raw=projmatrix_raw,
            sh_degree=int(getattr(pc, "active_sh_degree", 3)),
            campos=viewpoint_camera.camera_center.to(device=dev, dtype=torch.float32),
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
            shs = pc.get_features
        else:
            colors_precomp = override_color

        result_list = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        rendered_image, radii = result_list[0], result_list[1]

        if self.training:
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass

        return {
            "render": rendered_image.clamp(0, 1),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
