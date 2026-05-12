from __future__ import annotations

from dataclasses import dataclass

import torch

from rotationquant.metrics import distribution_metrics, tensor_metrics
from rotationquant.quantizers import (
    gaussian_lloyd_max_quantize,
    mxfp4_e2m1_quantize_flat_blocks,
    symmetric_absmax_quantize_flat_blocks,
    symmetric_absmax_quantize_last_dim_blocks,
)
from rotationquant.rotations import polar_rotation_forward, polar_rotation_inverse


@dataclass(frozen=True)
class StageAMethod:
    name: str
    rotation: str
    quantizer: str
    int_gemm_friendly: str
    rotation_backend: str | None = None
    allowed_bits: tuple[int, ...] | None = None


STAGE_A_METHODS: dict[str, StageAMethod] = {
    "direct_absmax": StageAMethod(
        name="direct_absmax",
        rotation="none",
        quantizer="block_absmax",
        int_gemm_friendly="Yes",
    ),
    "mxfp4": StageAMethod(
        name="mxfp4",
        rotation="none",
        quantizer="mxfp4_e2m1",
        int_gemm_friendly="Yes-ish; MXFP4-style block floating-point path",
        allowed_bits=(4,),
    ),
    "hadamard_absmax": StageAMethod(
        name="hadamard_absmax",
        rotation="polar_hadamard_blockwise",
        quantizer="block_absmax",
        int_gemm_friendly="Yes-ish; needs rotation handling",
        rotation_backend="hadamard",
    ),
    "hadamard_lm": StageAMethod(
        name="hadamard_lm",
        rotation="polar_hadamard_blockwise",
        quantizer="gaussian_lloyd_max",
        int_gemm_friendly="No; needs LUT/dequant or codebook-aware MAC",
        rotation_backend="hadamard",
    ),
    "hadamard_mxfp4": StageAMethod(
        name="hadamard_mxfp4",
        rotation="polar_hadamard_blockwise",
        quantizer="mxfp4_e2m1",
        int_gemm_friendly="Yes-ish; rotated MXFP4 needs rotation handling",
        rotation_backend="hadamard",
        allowed_bits=(4,),
    ),
    "randhadamard_lm": StageAMethod(
        name="randhadamard_lm",
        rotation="polar_randomized_hadamard_blockwise",
        quantizer="gaussian_lloyd_max",
        int_gemm_friendly="No; needs LUT/dequant or codebook-aware MAC",
        rotation_backend="randomized_hadamard",
    ),
    "randortho_lm": StageAMethod(
        name="randortho_lm",
        rotation="polar_random_orthogonal_blockwise",
        quantizer="gaussian_lloyd_max",
        int_gemm_friendly="No; dense random rotation is numeric-only in this experiment",
        rotation_backend="random_orthogonal",
    ),
}


def stage_a_method_supports_bits(method_name: str, bits: int) -> bool:
    method = STAGE_A_METHODS[method_name]
    return method.allowed_bits is None or bits in method.allowed_bits


def quantize_weight_for_stage_a(
    weight: torch.Tensor,
    bits: int,
    method_name: str,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, object]]:
    method = STAGE_A_METHODS[method_name]
    if not stage_a_method_supports_bits(method_name, bits):
        raise ValueError(f"Method {method_name} does not support W{bits}.")
    original_dtype = weight.dtype

    if method.rotation == "none":
        # Direct baseline uses local block scales so the baseline is comparable
        # with the block-wise rotated groups.
        if method.quantizer == "block_absmax":
            result = symmetric_absmax_quantize_flat_blocks(weight, bits, block_size=block_size)
        elif method.quantizer == "mxfp4_e2m1":
            result = mxfp4_e2m1_quantize_flat_blocks(weight, group_size=mxfp4_group_size)
        else:
            raise ValueError(f"Unsupported direct quantizer: {method.quantizer}")
        return result.values.to(dtype=original_dtype), {
            "method": method.name,
            "bits": bits,
            "rotation": method.rotation,
            "rotation_backend": "none",
            "quantizer_type": result.quantizer_type,
            "int_gemm_friendly": method.int_gemm_friendly,
            "block_size": block_size,
            "mxfp4_group_size": mxfp4_group_size if method.quantizer == "mxfp4_e2m1" else "",
            **result.metadata,
        }

    # Rotated groups: quantize the Gaussianized block space, then restore an
    # FP tensor so downstream layer/model code can still use normal matmul.
    rotation_backend = method.rotation_backend or "hadamard"
    rotated, norms, pad = polar_rotation_forward(
        weight,
        block_size=block_size,
        rotation_backend=rotation_backend,
        seed=rotation_seed,
    )
    if method.quantizer == "block_absmax":
        quantized = symmetric_absmax_quantize_last_dim_blocks(rotated, bits, block_size=block_size)
    elif method.quantizer == "gaussian_lloyd_max":
        quantized = gaussian_lloyd_max_quantize(rotated, bits)
    elif method.quantizer == "mxfp4_e2m1":
        quantized = mxfp4_e2m1_quantize_flat_blocks(rotated, group_size=mxfp4_group_size)
    else:
        raise ValueError(f"Unsupported rotated quantizer: {method.quantizer}")

    restored = polar_rotation_inverse(
        quantized.values,
        norms,
        shape=weight.shape,
        pad=pad,
        block_size=block_size,
        rotation_backend=rotation_backend,
        seed=rotation_seed,
    ).to(dtype=original_dtype)
    return restored, {
        "method": method.name,
        "bits": bits,
        "rotation": method.rotation,
        "rotation_backend": rotation_backend,
        "rotation_seed": rotation_seed,
        "quantizer_type": quantized.quantizer_type,
        "int_gemm_friendly": method.int_gemm_friendly,
        "block_size": block_size,
        "mxfp4_group_size": mxfp4_group_size if method.quantizer == "mxfp4_e2m1" else "",
        **quantized.metadata,
    }


def stage_a_tensor_record(
    layer_name: str,
    weight: torch.Tensor,
    bits: int,
    method_name: str,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
) -> dict[str, object]:
    quantized_weight, metadata = quantize_weight_for_stage_a(
        weight,
        bits=bits,
        method_name=method_name,
        block_size=block_size,
        mxfp4_group_size=mxfp4_group_size,
        rotation_seed=rotation_seed,
    )
    return {
        "layer": layer_name,
        **metadata,
        **tensor_metrics(weight, quantized_weight),
        **{f"weight_{k}": v for k, v in distribution_metrics(weight).items()},
    }
