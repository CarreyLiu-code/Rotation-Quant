from __future__ import annotations

from dataclasses import dataclass

import torch

from rotationquant.metrics import distribution_metrics, tensor_metrics
from rotationquant.quantizers import gaussian_lloyd_max_quantize, symmetric_absmax_quantize
from rotationquant.rotations import polar_hadamard_forward, polar_hadamard_inverse


@dataclass(frozen=True)
class StageAMethod:
    name: str
    rotation: str
    quantizer: str
    int_gemm_friendly: str


STAGE_A_METHODS: dict[str, StageAMethod] = {
    "direct_absmax": StageAMethod(
        name="direct_absmax",
        rotation="none",
        quantizer="absmax",
        int_gemm_friendly="Yes",
    ),
    "hadamard_absmax": StageAMethod(
        name="hadamard_absmax",
        rotation="polar_hadamard_blockwise",
        quantizer="absmax",
        int_gemm_friendly="Yes-ish; needs rotation handling",
    ),
    "hadamard_lm": StageAMethod(
        name="hadamard_lm",
        rotation="polar_hadamard_blockwise",
        quantizer="gaussian_lloyd_max",
        int_gemm_friendly="No; needs LUT/dequant or codebook-aware MAC",
    ),
}


def quantize_weight_for_stage_a(
    weight: torch.Tensor,
    bits: int,
    method_name: str,
    block_size: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    method = STAGE_A_METHODS[method_name]
    original_dtype = weight.dtype

    if method.rotation == "none":
        # Direct baseline: regular uniform fake quantization in the original
        # weight domain, matching the A1 group in the experiment plan.
        if method.quantizer != "absmax":
            raise ValueError(f"Unsupported direct quantizer: {method.quantizer}")
        result = symmetric_absmax_quantize(weight, bits)
        return result.values.to(dtype=original_dtype), {
            "method": method.name,
            "bits": bits,
            "rotation": method.rotation,
            "quantizer_type": result.quantizer_type,
            "int_gemm_friendly": method.int_gemm_friendly,
            **result.metadata,
        }

    # Rotated groups: quantize the Gaussianized block space, then restore an
    # FP tensor so downstream layer/model code can still use normal matmul.
    rotated, norms, pad = polar_hadamard_forward(weight, block_size=block_size)
    if method.quantizer == "absmax":
        quantized = symmetric_absmax_quantize(rotated, bits)
    elif method.quantizer == "gaussian_lloyd_max":
        quantized = gaussian_lloyd_max_quantize(rotated, bits)
    else:
        raise ValueError(f"Unsupported rotated quantizer: {method.quantizer}")

    restored = polar_hadamard_inverse(
        quantized.values,
        norms,
        shape=weight.shape,
        pad=pad,
        block_size=block_size,
    ).to(dtype=original_dtype)
    return restored, {
        "method": method.name,
        "bits": bits,
        "rotation": method.rotation,
        "quantizer_type": quantized.quantizer_type,
        "int_gemm_friendly": method.int_gemm_friendly,
        "block_size": block_size,
        **quantized.metadata,
    }


def stage_a_tensor_record(
    layer_name: str,
    weight: torch.Tensor,
    bits: int,
    method_name: str,
    block_size: int = 128,
) -> dict[str, object]:
    quantized_weight, metadata = quantize_weight_for_stage_a(
        weight,
        bits=bits,
        method_name=method_name,
        block_size=block_size,
    )
    return {
        "layer": layer_name,
        **metadata,
        **tensor_metrics(weight, quantized_weight),
        **{f"weight_{k}": v for k, v in distribution_metrics(weight).items()},
    }
