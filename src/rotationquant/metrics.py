from __future__ import annotations

import math

import torch


def relative_mse(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    ref = reference.float()
    cand = candidate.float()
    return float((ref - cand).square().mean().div(ref.square().mean().clamp_min(eps)).cpu())


def cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    denom = ref.norm().mul(cand.norm()).clamp_min(eps)
    return float(ref.dot(cand).div(denom).cpu())


def sqnr_db(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    ref = reference.float()
    noise = (ref - candidate.float()).square().mean().clamp_min(eps)
    signal = ref.square().mean().clamp_min(eps)
    return float((10.0 * torch.log10(signal / noise)).cpu())


def max_mean_ratio(x: torch.Tensor, eps: float = 1e-12) -> float:
    values = x.float().abs()
    return float(values.max().div(values.mean().clamp_min(eps)).cpu())


def kurtosis(x: torch.Tensor, eps: float = 1e-12) -> float:
    values = x.float().reshape(-1)
    centered = values - values.mean()
    var = centered.square().mean().clamp_min(eps)
    return float((centered.pow(4).mean() / var.square()).cpu())


def outlier_ratio(x: torch.Tensor, threshold: float = 3.0, eps: float = 1e-12) -> float:
    """Fraction of values farther than threshold standard deviations from mean."""
    values = x.float().reshape(-1)
    if values.numel() == 0:
        return math.nan
    centered = values - values.mean()
    std = centered.square().mean().sqrt().clamp_min(eps)
    return float((centered.abs() > threshold * std).float().mean().cpu())


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    return {
        "relative_mse": relative_mse(reference, candidate),
        "cosine": cosine_similarity(reference, candidate),
        "sqnr_db": sqnr_db(reference, candidate),
    }


def distribution_metrics(x: torch.Tensor) -> dict[str, float]:
    values = x.float()
    return {
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "max_mean_ratio": max_mean_ratio(values),
        "kurtosis": kurtosis(values),
        "outlier_ratio_3sigma": outlier_ratio(values, threshold=3.0),
        "numel": int(values.numel()),
        "nan_count": int(torch.isnan(values).sum().cpu()),
        "inf_count": int(torch.isinf(values).sum().cpu()),
        "finite_ratio": float(torch.isfinite(values).float().mean().cpu()) if values.numel() else math.nan,
    }
