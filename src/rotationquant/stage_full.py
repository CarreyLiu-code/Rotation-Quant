from __future__ import annotations

from dataclasses import dataclass

import torch

from rotationquant.stage_b import (
    STAGE_B_METHODS,
    STAGE_B_MODEL_METHODS,
    apply_stage_b_ffn_fake_quant_,
)
from rotationquant.stage_c import STAGE_C_KV_SPECS, STAGE_C_STRUCTURED_ATTENTION_SPECS
from rotationquant.stage_c_model import apply_stage_c_attention_fake_quant_


@dataclass(frozen=True)
class FullModelQuantSpec:
    name: str
    ffn_method: str | None
    attention_method: str | None
    group: str
    description: str


STAGE_FULL_MODEL_METHODS: dict[str, FullModelQuantSpec] = {
    "full_identity_attention_fp16": FullModelQuantSpec(
        name="full_identity_attention_fp16",
        ffn_method=None,
        attention_method="attn_identity_fp16",
        group="sanity",
        description="identity attention wrapper; FFN remains FP16",
    ),
    "full_direct_absmax_w4a4_absmax_k4v4": FullModelQuantSpec(
        name="full_direct_absmax_w4a4_absmax_k4v4",
        ffn_method="ffn_direct_absmax_w4a4",
        attention_method="attn_direct_absmax_w4a4_absmax_k4v4",
        group="4bit_baseline",
        description="block absmax W4A4 FFN/Attention with absmax K4V4",
    ),
    "full_mxfp4_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_mxfp4_w4a4_hlm_k4v4",
        ffn_method="ffn_mxfp4_w4a4",
        attention_method="attn_mxfp4_w4a4_hlm_k4v4",
        group="4bit_baseline",
        description="MXFP4 W4A4 FFN/Attention with HLM K4V4",
    ),
    "full_rot_absmax_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_rot_absmax_w4a4_hlm_k4v4",
        ffn_method="ffn_rot_absmax_w4a4",
        attention_method="attn_rot_absmax_w4a4_hlm_k4v4",
        group="4bit_baseline",
        description="Hadamard-rotated absmax W4A4 FFN/Attention with HLM K4V4",
    ),
    "full_rot_mxfp4_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_rot_mxfp4_w4a4_hlm_k4v4",
        ffn_method="ffn_rot_mxfp4_w4a4",
        attention_method="attn_rot_mxfp4_w4a4_hlm_k4v4",
        group="4bit_baseline",
        description="Hadamard rotation plus MXFP4 W4A4 FFN/Attention with HLM K4V4",
    ),
    "full_rot_lm_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_rot_lm_w4a4_hlm_k4v4",
        ffn_method="ffn_rot_lm_w4a4",
        attention_method="attn_rot_lm_w4a4_hlm_k4v4",
        group="4bit_main",
        description="Hadamard rotation plus Lloyd-Max W4A4 FFN/Attention with HLM K4V4",
    ),
    "full_randhadamard_lm_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_randhadamard_lm_w4a4_hlm_k4v4",
        ffn_method="ffn_randhadamard_lm_w4a4",
        attention_method="attn_randhadamard_lm_w4a4_hlm_k4v4",
        group="rotation_backend",
        description="randomized Hadamard Lloyd-Max W4A4 FFN/Attention with randomized HLM K4V4",
    ),
    "full_randortho_lm_w4a4_hlm_k4v4": FullModelQuantSpec(
        name="full_randortho_lm_w4a4_hlm_k4v4",
        ffn_method="ffn_randortho_lm_w4a4",
        attention_method="attn_randortho_lm_w4a4_hlm_k4v4",
        group="rotation_backend",
        description="dense random orthogonal Lloyd-Max W4A4 FFN/Attention with random-orthogonal K4V4",
    ),
    "full_rot_lm_w3a4_hlm_k3v4": FullModelQuantSpec(
        name="full_rot_lm_w3a4_hlm_k3v4",
        ffn_method="ffn_rot_lm_w3a4",
        attention_method="attn_rot_lm_w3a4_hlm_k3v4",
        group="low_bit",
        description="Hadamard Lloyd-Max W3A4 FFN/Attention with HLM K3V4",
    ),
    "full_rot_lm_w4a3_hlm_k4v3": FullModelQuantSpec(
        name="full_rot_lm_w4a3_hlm_k4v3",
        ffn_method="ffn_rot_lm_w4a3",
        attention_method="attn_rot_lm_w4a3_hlm_k4v3",
        group="low_bit",
        description="Hadamard Lloyd-Max W4A3 FFN/Attention with HLM K4V3",
    ),
    "full_rot_lm_w3a3_hlm_k3v3": FullModelQuantSpec(
        name="full_rot_lm_w3a3_hlm_k3v3",
        ffn_method="ffn_rot_lm_w3a3",
        attention_method="attn_rot_lm_w3a3_hlm_k3v3",
        group="low_bit",
        description="Hadamard Lloyd-Max W3A3 FFN/Attention with HLM K3V3",
    ),
    "full_rot_lm_w2a4_hlm_k2v4": FullModelQuantSpec(
        name="full_rot_lm_w2a4_hlm_k2v4",
        ffn_method="ffn_rot_lm_w2a4",
        attention_method="attn_rot_lm_w2a4_hlm_k2v4",
        group="failure_boundary",
        description="Hadamard Lloyd-Max W2A4 FFN/Attention with HLM K2V4",
    ),
    "full_mixed_ffn_w3a4_attn_w4a4_k4v4": FullModelQuantSpec(
        name="full_mixed_ffn_w3a4_attn_w4a4_k4v4",
        ffn_method="ffn_rot_lm_w3a4",
        attention_method="attn_rot_lm_w4a4_hlm_k4v4",
        group="mixed_precision",
        description="FFN Rot-LM W3A4 with Attention Rot-LM W4A4 and HLM K4V4",
    ),
    "full_mixed_ffn_w4a3_attn_w4a4_k4v4": FullModelQuantSpec(
        name="full_mixed_ffn_w4a3_attn_w4a4_k4v4",
        ffn_method="ffn_rot_lm_w4a3",
        attention_method="attn_rot_lm_w4a4_hlm_k4v4",
        group="mixed_precision",
        description="FFN Rot-LM W4A3 with Attention Rot-LM W4A4 and HLM K4V4",
    ),
    "full_mixed_ffn_w4a4_attn_w3a4_k3v4": FullModelQuantSpec(
        name="full_mixed_ffn_w4a4_attn_w3a4_k3v4",
        ffn_method="ffn_rot_lm_w4a4",
        attention_method="attn_rot_lm_w3a4_hlm_k3v4",
        group="mixed_precision",
        description="FFN Rot-LM W4A4 with Attention Rot-LM W3A4 and HLM K3V4",
    ),
    "full_rot_lm_w4a4_hlm_k4v4_h32": FullModelQuantSpec(
        name="full_rot_lm_w4a4_hlm_k4v4_h32",
        ffn_method="ffn_rot_lm_w4a4",
        attention_method="attn_rot_lm_w4a4_hlm_k4v4_h32",
        group="kv_block_size",
        description="Hadamard Lloyd-Max W4A4 with head-internal H32 HLM K4V4",
    ),
}


def apply_stage_full_fake_quant_(
    model: torch.nn.Module,
    method_name: str,
    *,
    block_size: int = 64,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
) -> dict[str, object]:
    """Apply one full-model fake quant method to a LLaMA-like model in-place."""
    if method_name not in STAGE_FULL_MODEL_METHODS:
        raise ValueError(f"Unknown Stage 2 full-model method: {method_name}")

    spec = STAGE_FULL_MODEL_METHODS[method_name]
    attention_records: list[dict[str, object]] = []
    ffn_records: list[dict[str, object]] = []

    # Attention is replaced first, then FFN. The two wrappers own disjoint
    # decoder-layer submodules and leave the residual stream unchanged.
    if spec.attention_method is not None:
        if spec.attention_method not in STAGE_C_STRUCTURED_ATTENTION_SPECS:
            raise ValueError(f"Unknown Stage C attention method: {spec.attention_method}")
        attention_records = apply_stage_c_attention_fake_quant_(
            model,
            spec.attention_method,
            block_size=block_size,
            mxfp4_group_size=mxfp4_group_size,
            rotation_seed=rotation_seed,
        )

    if spec.ffn_method is not None:
        if spec.ffn_method not in STAGE_B_MODEL_METHODS:
            raise ValueError(f"Unknown Stage B FFN method: {spec.ffn_method}")
        ffn_records = apply_stage_b_ffn_fake_quant_(
            model,
            spec.ffn_method,
            block_size=block_size,
            mxfp4_group_size=mxfp4_group_size,
            rotation_seed=rotation_seed,
        )

    return {
        "method_key": method_name,
        "method": spec.name,
        "group": spec.group,
        "description": spec.description,
        "ffn_method": spec.ffn_method or "fp16",
        "attention_method": spec.attention_method or "fp16",
        "quantized_attention_modules": len(attention_records),
        "quantized_ffn_modules": len(ffn_records),
        "attention_records": attention_records,
        "ffn_records": ffn_records,
    }


def stage_full_method_metadata(method_name: str) -> dict[str, object]:
    """Return compact method metadata for CSV and summary outputs."""
    if method_name == "fp16":
        return {
            "method": "fp16",
            "group": "baseline",
            "description": "baseline",
            "ffn_method": "fp16",
            "ffn_bits": "FP16",
            "ffn_quantizer": "none",
            "ffn_rotation_backend": "none",
            "attention_method": "fp16",
            "attention_linear_bits": "FP16",
            "attention_linear_quantizer": "none",
            "attention_rotation_backend": "none",
            "kv_method": "fp16",
            "kv_bits": "K16V16",
            "k_bits": 16,
            "v_bits": 16,
            "kv_rotation_backend": "none",
            "kv_block_size": "",
            "value_path": "reference",
        }

    spec = STAGE_FULL_MODEL_METHODS[method_name]
    ffn_bits = "FP16"
    ffn_quantizer = "none"
    ffn_rotation_backend = "none"
    if spec.ffn_method is not None:
        ffn_spec = STAGE_B_MODEL_METHODS[spec.ffn_method]
        ffn_method = STAGE_B_METHODS[ffn_spec.method]
        ffn_bits = ffn_spec.label
        ffn_quantizer = ffn_method.quantizer
        ffn_rotation_backend = ffn_method.rotation_backend or "none"

    attention_linear_bits = "FP16"
    attention_linear_quantizer = "none"
    attention_rotation_backend = "none"
    kv_method = "fp16"
    kv_bits = "K16V16"
    k_bits = 16
    v_bits = 16
    kv_rotation_backend = "none"
    kv_block_size: int | str = ""
    value_path = "reference"
    if spec.attention_method is not None:
        attention_spec = STAGE_C_STRUCTURED_ATTENTION_SPECS[spec.attention_method]
        kv_spec = STAGE_C_KV_SPECS[attention_spec.kv_spec_key]
        if attention_spec.linear_spec is not None:
            linear_method = STAGE_B_METHODS[attention_spec.linear_spec.method]
            attention_linear_bits = attention_spec.linear_spec.label
            attention_linear_quantizer = linear_method.quantizer
            attention_rotation_backend = linear_method.rotation_backend or "none"
        kv_method = kv_spec.method
        kv_bits = kv_spec.label
        k_bits = kv_spec.k_bits
        v_bits = kv_spec.v_bits
        kv_rotation_backend = kv_spec.rotation_backend
        kv_block_size = kv_spec.kv_block_size
        value_path = attention_spec.value_path

    return {
        "method": spec.name,
        "group": spec.group,
        "description": spec.description,
        "ffn_method": spec.ffn_method or "fp16",
        "ffn_bits": ffn_bits,
        "ffn_quantizer": ffn_quantizer,
        "ffn_rotation_backend": ffn_rotation_backend,
        "attention_method": spec.attention_method or "fp16",
        "attention_linear_bits": attention_linear_bits,
        "attention_linear_quantizer": attention_linear_quantizer,
        "attention_rotation_backend": attention_rotation_backend,
        "kv_method": kv_method,
        "kv_bits": kv_bits,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "kv_rotation_backend": kv_rotation_backend,
        "kv_block_size": kv_block_size,
        "value_path": value_path,
    }
