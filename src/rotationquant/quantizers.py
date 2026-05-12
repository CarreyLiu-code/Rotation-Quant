from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    bits: int
    quantizer_type: str
    metadata: dict[str, float | str | int]


def symmetric_absmax_quantize(x: torch.Tensor, bits: int, eps: float = 1e-12) -> QuantizedTensor:
    """Symmetric uniform fake quantization with dequantized float output."""
    if bits < 2:
        raise ValueError("symmetric_absmax_quantize expects bits >= 2.")
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().max().float().clamp_min(eps) / qmax
    q = torch.round(x.float() / scale).clamp(-qmax, qmax)
    dequant = (q * scale).to(dtype=x.dtype)
    return QuantizedTensor(
        values=dequant,
        bits=bits,
        quantizer_type="uniform integer-like",
        metadata={"scale": float(scale.cpu()), "qmax": qmax},
    )


def _check_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")


def _symmetric_absmax_quantize_blocks(blocks: torch.Tensor, bits: int, eps: float) -> tuple[torch.Tensor, int]:
    if bits < 2:
        raise ValueError("block absmax quantization expects bits >= 2.")
    qmax = (1 << (bits - 1)) - 1
    scale = blocks.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(eps) / qmax
    q = torch.round(blocks.float() / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=blocks.dtype), qmax


def symmetric_absmax_quantize_flat_blocks(
    x: torch.Tensor,
    bits: int,
    block_size: int = 128,
    eps: float = 1e-12,
) -> QuantizedTensor:
    """Symmetric uniform fake quantization with one absmax scale per flat block."""
    _check_block_size(block_size)
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    quantized, qmax = _symmetric_absmax_quantize_blocks(blocks, bits, eps)
    restored = quantized.reshape(-1)
    if pad:
        restored = restored[:-pad]
    return QuantizedTensor(
        values=restored.reshape_as(x),
        bits=bits,
        quantizer_type="uniform integer-like",
        metadata={
            "qmax": qmax,
            "block_size": block_size,
            "scale_granularity": "flat_block_absmax",
            "scale_count": int(blocks.shape[0]),
        },
    )


def symmetric_absmax_quantize_last_dim_blocks(
    x: torch.Tensor,
    bits: int,
    block_size: int = 128,
    eps: float = 1e-12,
) -> QuantizedTensor:
    """Symmetric uniform fake quantization with one absmax scale per last-dim block."""
    _check_block_size(block_size)
    pad = (-x.shape[-1]) % block_size
    values = x
    if pad:
        values = F.pad(values, (0, pad))
    leading_shape = values.shape[:-1]
    blocks = values.reshape(*leading_shape, values.shape[-1] // block_size, block_size)
    quantized, qmax = _symmetric_absmax_quantize_blocks(blocks, bits, eps)
    restored = quantized.reshape(*leading_shape, values.shape[-1])
    if pad:
        restored = restored[..., :-pad]
    return QuantizedTensor(
        values=restored.to(dtype=x.dtype),
        bits=bits,
        quantizer_type="uniform integer-like",
        metadata={
            "qmax": qmax,
            "block_size": block_size,
            "scale_granularity": "last_dim_block_absmax",
            "scale_count": int(blocks.numel() // block_size),
        },
    )


def _mxfp4_quantize_blocks(blocks: torch.Tensor, eps: float) -> torch.Tensor:
    codebook = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=blocks.device,
        dtype=torch.float32,
    )
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    amax = blocks.detach().abs().amax(dim=-1, keepdim=True).float()
    raw_scale = (amax / 6.0).clamp_min(eps)
    scale = torch.pow(torch.tensor(2.0, device=blocks.device), torch.ceil(torch.log2(raw_scale)))
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    normalized = blocks.float() / scale
    indices = torch.zeros_like(normalized, dtype=torch.long)
    for boundary in boundaries:
        indices = indices + (normalized > boundary).to(torch.long)
    return (codebook[indices] * scale).to(dtype=blocks.dtype)


def mxfp4_e2m1_quantize_flat_blocks(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-12,
) -> QuantizedTensor:
    """MXFP4 E2M1 fake quantization with power-of-two scale per flat group."""
    _check_block_size(group_size)
    flat = x.reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.reshape(-1, group_size)
    quantized = _mxfp4_quantize_blocks(blocks, eps)
    restored = quantized.reshape(-1)
    if pad:
        restored = restored[:-pad]
    return QuantizedTensor(
        values=restored.reshape_as(x),
        bits=4,
        quantizer_type="mxfp4_e2m1_fake_quant",
        metadata={
            "mxfp4_group_size": group_size,
            "scale_granularity": "group_power2",
            "scale_count": int(blocks.shape[0]),
            "element_format": "E2M1",
            "scale_format": "E8M0_like_power2",
        },
    )


def mxfp4_e2m1_quantize_last_dim_blocks(
    x: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-12,
) -> QuantizedTensor:
    """MXFP4 E2M1 fake quantization with power-of-two scale per last-dim group."""
    _check_block_size(group_size)
    pad = (-x.shape[-1]) % group_size
    values = x
    if pad:
        values = F.pad(values, (0, pad))
    leading_shape = values.shape[:-1]
    blocks = values.reshape(*leading_shape, values.shape[-1] // group_size, group_size)
    quantized = _mxfp4_quantize_blocks(blocks, eps)
    restored = quantized.reshape(*leading_shape, values.shape[-1])
    if pad:
        restored = restored[..., :-pad]
    return QuantizedTensor(
        values=restored.to(dtype=x.dtype),
        bits=4,
        quantizer_type="mxfp4_e2m1_fake_quant",
        metadata={
            "mxfp4_group_size": group_size,
            "scale_granularity": "group_power2",
            "scale_count": int(blocks.numel() // group_size),
            "element_format": "E2M1",
            "scale_format": "E8M0_like_power2",
        },
    )


@lru_cache(maxsize=16)
def gaussian_lloyd_max_codebook(bits: int, grid_size: int = 20001, iters: int = 80) -> tuple[float, ...]:
    """Numerically fit a Lloyd-Max codebook for N(0, 1)."""
    if bits < 1:
        raise ValueError("Lloyd-Max bits must be positive.")
    levels = 1 << bits
    grid = torch.linspace(-8.0, 8.0, grid_size, dtype=torch.float64)
    pdf = torch.exp(-0.5 * grid.square())
    pdf = pdf / pdf.sum()

    # Fit on a dense finite grid rather than depending on scipy. This keeps the
    # codebook deterministic and cheap to cache for the small bit-widths used here.
    centroids = torch.linspace(-2.5, 2.5, levels, dtype=torch.float64)
    for _ in range(iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        indices = torch.bucketize(grid, boundaries)
        updated = centroids.clone()
        for i in range(levels):
            mask = indices == i
            if mask.any():
                mass = pdf[mask].sum().clamp_min(1e-30)
                updated[i] = (grid[mask] * pdf[mask]).sum() / mass
        if torch.max(torch.abs(updated - centroids)) < 1e-10:
            centroids = updated
            break
        centroids = updated
    return tuple(float(v) for v in centroids)


def gaussian_lloyd_max_quantize(x: torch.Tensor, bits: int) -> QuantizedTensor:
    """Standard-normal Lloyd-Max fake quantization with centroid dequantization."""
    codebook = torch.tensor(
        gaussian_lloyd_max_codebook(bits),
        device=x.device,
        dtype=torch.float32,
    )
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    # This is fake quant: values come back as centroids, not integer codes.
    indices = torch.bucketize(x.float(), boundaries)
    dequant = codebook[indices].to(dtype=x.dtype)
    return QuantizedTensor(
        values=dequant,
        bits=bits,
        quantizer_type="non-uniform codebook",
        metadata={"levels": len(codebook), "codebook": "gaussian_lloyd_max_standard_normal"},
    )
