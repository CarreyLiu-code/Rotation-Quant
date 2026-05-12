from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _check_power_of_two(n: int) -> None:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Expected a power-of-two size, got {n}.")


def fwht(x: torch.Tensor, dim: int = -1, normalize: bool = True) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along one dimension."""
    n = x.shape[dim]
    _check_power_of_two(n)

    y = x.movedim(dim, -1).contiguous()
    original_shape = y.shape
    y = y.reshape(-1, n)

    # Iterative butterfly schedule. Each pass combines neighboring chunks of
    # length h into Hadamard pairs, avoiding materializing an H_n matrix.
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.stack((left + right, left - right), dim=2)
        h *= 2
        y = y.reshape(-1, n)

    if normalize:
        y = y / math.sqrt(n)
    return y.reshape(original_shape).movedim(-1, dim)


def random_signs(
    size: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a deterministic Rademacher sign vector for randomized Hadamard."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
    return signs.mul(2).sub(1).to(device=device, dtype=dtype)


def random_orthogonal_matrix(
    size: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a deterministic dense orthogonal matrix from a Gaussian QR draw."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    diag = torch.sign(torch.diag(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q * diag
    return q.to(device=device, dtype=dtype)


def apply_rotation_last_dim(
    x: torch.Tensor,
    *,
    rotation_backend: str = "hadamard",
    seed: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply a selected orthonormal rotation along the last dimension."""
    size = x.shape[-1]
    if rotation_backend == "hadamard":
        return fwht(x, dim=-1, normalize=True)
    if rotation_backend == "randomized_hadamard":
        signs = random_signs(size, seed=seed, device=x.device, dtype=x.dtype)
        if inverse:
            return fwht(x, dim=-1, normalize=True) * signs
        return fwht(x * signs, dim=-1, normalize=True)
    if rotation_backend == "random_orthogonal":
        matrix = random_orthogonal_matrix(size, seed=seed, device=x.device, dtype=x.dtype)
        return x @ (matrix.t() if inverse else matrix)
    raise ValueError(f"Unsupported rotation_backend: {rotation_backend}")


def block_rotation_last_dim(
    x: torch.Tensor,
    block_size: int = 128,
    *,
    rotation_backend: str = "hadamard",
    seed: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply an orthonormal rotation independently to last-dim blocks."""
    _check_power_of_two(block_size)
    pad = (-x.shape[-1]) % block_size
    values = F.pad(x, (0, pad)) if pad else x
    leading_shape = values.shape[:-1]
    blocks = values.reshape(*leading_shape, values.shape[-1] // block_size, block_size)
    rotated = apply_rotation_last_dim(
        blocks,
        rotation_backend=rotation_backend,
        seed=seed,
        inverse=inverse,
    ).reshape(*leading_shape, values.shape[-1])
    if pad:
        rotated = rotated[..., :-pad]
    return rotated


def block_view_1d(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """Flatten a tensor and pad it to full blocks."""
    _check_power_of_two(block_size)
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.reshape(-1, block_size), pad


def restore_block_view(blocks: torch.Tensor, shape: torch.Size, pad: int) -> torch.Tensor:
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(shape)


def normalize_blocks(blocks: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    norms = blocks.float().norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return blocks.float() / norms, norms


def polar_hadamard_forward(
    weight: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """PolarQuant-style block normalization followed by orthonormal Hadamard."""
    blocks, pad = block_view_1d(weight, block_size)
    normalized, norms = normalize_blocks(blocks, eps=eps)
    rotated = fwht(normalized, dim=-1, normalize=True)
    # After L2 normalization, sqrt(d) * Hx is approximately standard normal
    # under the Gaussianization argument used by PolarQuant/TurboQuant.
    gaussianized = rotated * math.sqrt(block_size)
    return gaussianized, norms, pad


def polar_hadamard_inverse(
    gaussianized: torch.Tensor,
    norms: torch.Tensor,
    shape: torch.Size,
    pad: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Invert polar_hadamard_forward after quantizing the Gaussianized values."""
    rotated = gaussianized / math.sqrt(block_size)
    # Orthonormal Hadamard is self-inverse, so the same FWHT restores blocks.
    normalized_hat = fwht(rotated, dim=-1, normalize=True)
    blocks_hat = normalized_hat * norms
    return restore_block_view(blocks_hat, shape, pad)


def polar_rotation_forward(
    weight: torch.Tensor,
    block_size: int = 128,
    *,
    rotation_backend: str = "hadamard",
    seed: int = 0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Block normalization followed by a selected orthonormal rotation."""
    blocks, pad = block_view_1d(weight, block_size)
    normalized, norms = normalize_blocks(blocks, eps=eps)
    rotated = apply_rotation_last_dim(
        normalized,
        rotation_backend=rotation_backend,
        seed=seed,
        inverse=False,
    )
    return rotated * math.sqrt(block_size), norms, pad


def polar_rotation_inverse(
    gaussianized: torch.Tensor,
    norms: torch.Tensor,
    shape: torch.Size,
    pad: int,
    block_size: int = 128,
    *,
    rotation_backend: str = "hadamard",
    seed: int = 0,
) -> torch.Tensor:
    """Invert polar_rotation_forward after quantizing the Gaussianized values."""
    rotated = gaussianized / math.sqrt(block_size)
    normalized_hat = apply_rotation_last_dim(
        rotated,
        rotation_backend=rotation_backend,
        seed=seed,
        inverse=True,
    )
    blocks_hat = normalized_hat * norms
    return restore_block_view(blocks_hat, shape, pad)
