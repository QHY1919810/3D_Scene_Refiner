# -*- coding: utf-8 -*-
"""
Supervised losses for observed-view training (threestudio-3dgs)

This version is ROBUST to resolution mismatch between:
- renderer output pred_rgb: [B,H,W,3]
- GT rgb/mask loaded from disk: [B,h,w,3]/[B,h,w,1]

It automatically resizes GT rgb (bilinear) and mask (nearest) to match pred resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _resize_bhwc(x: torch.Tensor, size_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    """Resize BHWC tensor to (H,W) using interpolate on BCHW."""
    assert x.ndim == 4, f"expected 4D tensor, got {tuple(x.shape)}"
    b, h, w, c = x.shape
    H, W = size_hw
    if (h, w) == (H, W):
        return x
    x_bchw = x.permute(0, 3, 1, 2)
    if mode in ("bilinear", "bicubic"):
        x_rs = F.interpolate(x_bchw, size=(H, W), mode=mode, align_corners=False)
    else:
        x_rs = F.interpolate(x_bchw, size=(H, W), mode=mode)
    return x_rs.permute(0, 2, 3, 1).contiguous()


def _ensure_mask_bhw1(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    return mask[..., :1]


def _match_to_pred(
    pred_bhwc: torch.Tensor,
    gt_bhwc: Optional[torch.Tensor],
    mask_bhw1: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Resize gt/mask to match pred size (if provided)."""
    H, W = int(pred_bhwc.shape[1]), int(pred_bhwc.shape[2])
    if gt_bhwc is not None and (gt_bhwc.shape[1] != H or gt_bhwc.shape[2] != W):
        gt_bhwc = _resize_bhwc(gt_bhwc, (H, W), mode="bilinear")
    if mask_bhw1 is not None:
        mask_bhw1 = _ensure_mask_bhw1(mask_bhw1)
        if mask_bhw1.shape[1] != H or mask_bhw1.shape[2] != W:
            mask_bhw1 = _resize_bhwc(mask_bhw1, (H, W), mode="nearest")
    return pred_bhwc, gt_bhwc, mask_bhw1


def _dilate_mask(mask_bhw1: torch.Tensor, mask_dilate: int) -> torch.Tensor:
    if mask_dilate <= 0:
        return mask_bhw1
    m = mask_bhw1.permute(0, 3, 1, 2)  # B,1,H,W
    m = F.max_pool2d(m, kernel_size=mask_dilate * 2 + 1, stride=1, padding=mask_dilate)
    return m.permute(0, 2, 3, 1)


def masked_l1(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: Optional[torch.Tensor],
    mask_dilate: int = 0,
) -> torch.Tensor:
    pred_rgb, gt_rgb, mask = _match_to_pred(pred_rgb, gt_rgb, mask)
    diff = (pred_rgb - gt_rgb).abs()
    if mask is None:
        return diff.mean()
    mask = _dilate_mask(mask.to(device=diff.device, dtype=diff.dtype), mask_dilate)
    w = mask.expand_as(diff)
    return (diff * w).sum() / (w.sum() + 1e-8)


def masked_l2(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: Optional[torch.Tensor],
    mask_dilate: int = 0,
) -> torch.Tensor:
    pred_rgb, gt_rgb, mask = _match_to_pred(pred_rgb, gt_rgb, mask)
    diff = (pred_rgb - gt_rgb)
    diff = diff * diff
    if mask is None:
        return diff.mean()
    mask = _dilate_mask(mask.to(device=diff.device, dtype=diff.dtype), mask_dilate)
    w = mask.expand_as(diff)
    return (diff * w).sum() / (w.sum() + 1e-8)


def silhouette_bce(
    pred_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    # pred_alpha: [B,H,W,1] or [B,H,W]
    if pred_alpha.ndim == 3:
        pred_alpha = pred_alpha.unsqueeze(-1)
    pred_alpha = pred_alpha[..., :1]
    # match sizes
    pred_alpha, _, gt_mask = _match_to_pred(pred_alpha, None, gt_mask)
    gt_mask = _ensure_mask_bhw1(gt_mask)
    pred_alpha = pred_alpha.clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(pred_alpha, gt_mask.to(device=pred_alpha.device, dtype=pred_alpha.dtype))


def compute_sup_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_rgb: float = 1.0,
    lambda_silhouette: float = 0.0,
    rgb_loss_type: str = "l1",
    mask_dilate: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    outputs: expects outputs["comp_rgb"] [B,H,W,3], optionally outputs["comp_mask"] [B,H,W,1]
    batch: expects batch["rgb"] [B,h,w,3], optionally batch["mask"] [B,h,w,1]
    """
    pred_rgb = outputs["comp_rgb"]
    gt_rgb = batch["rgb"].to(device=pred_rgb.device, dtype=pred_rgb.dtype)
    gt_mask = batch.get("mask", None)
    if gt_mask is not None:
        gt_mask = gt_mask.to(device=pred_rgb.device, dtype=pred_rgb.dtype)

    rgb_loss_type = str(rgb_loss_type).lower()
    if rgb_loss_type in ("l2", "mse"):
        loss_rgb = masked_l2(pred_rgb, gt_rgb, gt_mask, mask_dilate=mask_dilate)
    else:
        loss_rgb = masked_l1(pred_rgb, gt_rgb, gt_mask, mask_dilate=mask_dilate)

    loss = lambda_rgb * loss_rgb

    loss_sil = pred_rgb.sum() * 0.0
    if lambda_silhouette > 0 and ("comp_mask" in outputs) and (gt_mask is not None):
        loss_sil = silhouette_bce(outputs["comp_mask"], gt_mask)
        loss = loss + lambda_silhouette * loss_sil

    return {"loss_sup": loss, "loss_rgb": loss_rgb, "loss_silhouette": loss_sil}
