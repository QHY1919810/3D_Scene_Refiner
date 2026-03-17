import torch
import torch.nn.functional as F
from typing import Optional, Literal

HFMode = Literal["laplacian", "sobel"]
HFLossType = Literal["l1", "l2"]

_LAPLACIAN_4 = torch.tensor([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=torch.float32)

_SOBEL_X = torch.tensor([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=torch.float32)

_SOBEL_Y = torch.tensor([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=torch.float32)


def _to_bchw(rgb_hwc: torch.Tensor) -> torch.Tensor:
    # rgb_hwc: [B,H,W,C]
    return rgb_hwc.permute(0, 3, 1, 2).contiguous()


def _to_gray(bchw: torch.Tensor) -> torch.Tensor:
    # bchw: [B,3,H,W] -> [B,1,H,W]
    if bchw.shape[1] == 1:
        return bchw
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=bchw.device, dtype=bchw.dtype).view(1, 3, 1, 1)
    return (bchw * w).sum(dim=1, keepdim=True)


def high_frequency_map(
    rgb: torch.Tensor,
    mode: HFMode = "laplacian",
    grayscale: bool = True,
) -> torch.Tensor:
    """Compute high-frequency response map.
    rgb: [B,H,W,3] float. returns [B,1,H,W] if grayscale else [B,C,H,W]
    """
    x = _to_bchw(rgb)
    if grayscale:
        x = _to_gray(x)

    B, C, H, W = x.shape
    if mode == "laplacian":
        k = _LAPLACIAN_4.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        return F.conv2d(x, k, padding=1, groups=C)

    if mode == "sobel":
        kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    raise ValueError(f"Unknown mode: {mode}")


def _dilate_mask(mask_b1hw: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask_b1hw
    return F.max_pool2d(mask_b1hw, kernel_size=radius * 2 + 1, stride=1, padding=radius)


def masked_high_frequency_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mode: HFMode = "laplacian",
    loss_type: HFLossType = "l1",
    grayscale: bool = True,
    mask_dilate: int = 0,
) -> torch.Tensor:
    """Masked HF loss between pred and gt.
    pred_rgb, gt_rgb: [B,H,W,3]
    mask: [B,H,W,1] or [B,H,W] (1=keep). If None -> unmasked.
    """
    assert pred_rgb.shape == gt_rgb.shape, f"pred {pred_rgb.shape} != gt {gt_rgb.shape}"
    hf_p = high_frequency_map(pred_rgb, mode=mode, grayscale=grayscale)
    hf_g = high_frequency_map(gt_rgb, mode=mode, grayscale=grayscale)

    diff = (hf_p - hf_g).abs() if loss_type == "l1" else (hf_p - hf_g) ** 2

    if mask is None:
        return diff.mean()

    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    mask = mask[..., :1]
    m = mask.permute(0, 3, 1, 2).to(device=diff.device, dtype=diff.dtype)
    m = _dilate_mask(m, mask_dilate)

    w = m.expand(diff.shape[0], diff.shape[1], diff.shape[2], diff.shape[3])
    denom = w.sum() + 1e-8
    return (diff * w).sum() / denom
