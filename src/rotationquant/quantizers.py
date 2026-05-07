from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


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
