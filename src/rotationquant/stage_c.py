from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotationquant.metrics import cosine_similarity, relative_mse, tensor_metrics
from rotationquant.quantizers import gaussian_lloyd_max_codebook
from rotationquant.rotations import block_rotation_last_dim, fwht, random_orthogonal_matrix
from rotationquant.stage_b import STAGE_B_LINEAR_SPECS, WABitSpec


@dataclass(frozen=True)
class KVQuantSpec:
    name: str
    method: str
    k_bits: int
    v_bits: int
    rotation: str
    quantizer: str
    compute_interpretation: str
    rotation_backend: str = "hadamard"
    kv_block_size: int = 64

    @property
    def label(self) -> str:
        return f"K{self.k_bits}V{self.v_bits}"


@dataclass(frozen=True)
class QJLSpec:
    name: str
    base_k_bits: int
    residual_bits: int
    projection_dim: int
    value_spec_key: str
    compute_interpretation: str

    @property
    def label(self) -> str:
        suffix = "+QJL" if self.residual_bits else ""
        return f"K{self.base_k_bits}{suffix}"


@dataclass(frozen=True)
class StageCStructuredAttentionSpec:
    name: str
    linear_spec: WABitSpec | None
    kv_spec_key: str
    value_path: str
    quantize_qkv: bool
    quantize_o: bool
    compute_interpretation: str


@dataclass
class AttentionComputation:
    raw_inner_product: torch.Tensor
    scores: torch.Tensor
    probs: torch.Tensor
    output_heads: torch.Tensor
    key_hat: torch.Tensor
    value_hat: torch.Tensor
    metadata: dict[str, object]


STAGE_C_KV_SPECS: dict[str, KVQuantSpec] = {
    "fp16": KVQuantSpec(
        name="fp16",
        method="fp16",
        k_bits=16,
        v_bits=16,
        rotation="none",
        quantizer="none",
        compute_interpretation="baseline",
        rotation_backend="none",
    ),
    "absmax_k4v4": KVQuantSpec(
        name="absmax_k4v4",
        method="absmax",
        k_bits=4,
        v_bits=4,
        rotation="none",
        quantizer="absmax_per_token_head",
        compute_interpretation="uniform fake quant KV baseline",
        rotation_backend="none",
    ),
    "absmax_k3v4": KVQuantSpec(
        name="absmax_k3v4",
        method="absmax",
        k_bits=3,
        v_bits=4,
        rotation="none",
        quantizer="absmax_per_token_head",
        compute_interpretation="uniform fake quant with lower key bits",
        rotation_backend="none",
    ),
    "absmax_k4v3": KVQuantSpec(
        name="absmax_k4v3",
        method="absmax",
        k_bits=4,
        v_bits=3,
        rotation="none",
        quantizer="absmax_per_token_head",
        compute_interpretation="uniform fake quant with lower value bits",
        rotation_backend="none",
    ),
    "hadamard_lm_k4v4": KVQuantSpec(
        name="hadamard_lm_k4v4",
        method="hadamard_lm",
        k_bits=4,
        v_bits=4,
        rotation="headwise_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="rotated non-uniform KV fake quant",
        rotation_backend="hadamard",
    ),
    "hadamard_lm_k4v4_h32": KVQuantSpec(
        name="hadamard_lm_k4v4_h32",
        method="hadamard_lm",
        k_bits=4,
        v_bits=4,
        rotation="headwise_hadamard_h32",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="head-internal H32 non-uniform KV fake quant",
        rotation_backend="hadamard",
        kv_block_size=32,
    ),
    "hadamard_lm_k3v4": KVQuantSpec(
        name="hadamard_lm_k3v4",
        method="hadamard_lm",
        k_bits=3,
        v_bits=4,
        rotation="headwise_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="rotated non-uniform key bit reduction",
        rotation_backend="hadamard",
    ),
    "hadamard_lm_k3v4_h32": KVQuantSpec(
        name="hadamard_lm_k3v4_h32",
        method="hadamard_lm",
        k_bits=3,
        v_bits=4,
        rotation="headwise_hadamard_h32",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="head-internal H32 key bit reduction",
        rotation_backend="hadamard",
        kv_block_size=32,
    ),
    "hadamard_lm_k4v3": KVQuantSpec(
        name="hadamard_lm_k4v3",
        method="hadamard_lm",
        k_bits=4,
        v_bits=3,
        rotation="headwise_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="rotated non-uniform value bit reduction",
        rotation_backend="hadamard",
    ),
    "hadamard_lm_k3v3": KVQuantSpec(
        name="hadamard_lm_k3v3",
        method="hadamard_lm",
        k_bits=3,
        v_bits=3,
        rotation="headwise_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="aggressive rotated non-uniform KV fake quant",
        rotation_backend="hadamard",
    ),
    "hadamard_lm_k2v4": KVQuantSpec(
        name="hadamard_lm_k2v4",
        method="hadamard_lm",
        k_bits=2,
        v_bits=4,
        rotation="headwise_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="key 2-bit failure boundary",
        rotation_backend="hadamard",
    ),
    "randhadamard_lm_k4v4": KVQuantSpec(
        name="randhadamard_lm_k4v4",
        method="hadamard_lm",
        k_bits=4,
        v_bits=4,
        rotation="headwise_randomized_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="randomized Hadamard non-uniform KV fake quant",
        rotation_backend="randomized_hadamard",
    ),
    "randhadamard_lm_k3v4": KVQuantSpec(
        name="randhadamard_lm_k3v4",
        method="hadamard_lm",
        k_bits=3,
        v_bits=4,
        rotation="headwise_randomized_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="randomized Hadamard key bit reduction",
        rotation_backend="randomized_hadamard",
    ),
    "randhadamard_lm_k4v3": KVQuantSpec(
        name="randhadamard_lm_k4v3",
        method="hadamard_lm",
        k_bits=4,
        v_bits=3,
        rotation="headwise_randomized_hadamard",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="randomized Hadamard value bit reduction",
        rotation_backend="randomized_hadamard",
    ),
    "randortho_lm_k4v4": KVQuantSpec(
        name="randortho_lm_k4v4",
        method="hadamard_lm",
        k_bits=4,
        v_bits=4,
        rotation="headwise_random_orthogonal",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="dense random orthogonal non-uniform KV fake quant",
        rotation_backend="random_orthogonal",
    ),
    "randortho_lm_k3v4": KVQuantSpec(
        name="randortho_lm_k3v4",
        method="hadamard_lm",
        k_bits=3,
        v_bits=4,
        rotation="headwise_random_orthogonal",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="dense random orthogonal key bit reduction",
        rotation_backend="random_orthogonal",
    ),
    "randortho_lm_k4v3": KVQuantSpec(
        name="randortho_lm_k4v3",
        method="hadamard_lm",
        k_bits=4,
        v_bits=3,
        rotation="headwise_random_orthogonal",
        quantizer="gaussian_lloyd_max",
        compute_interpretation="dense random orthogonal value bit reduction",
        rotation_backend="random_orthogonal",
    ),
}


STAGE_C_QJL_SPECS: dict[str, QJLSpec] = {
    "hadamard_lm_k3": QJLSpec(
        name="hadamard_lm_k3",
        base_k_bits=3,
        residual_bits=0,
        projection_dim=64,
        value_spec_key="hadamard_lm_k4v4",
        compute_interpretation="pure Hadamard-LM key reconstruction",
    ),
    "hadamard_lm_k2": QJLSpec(
        name="hadamard_lm_k2",
        base_k_bits=2,
        residual_bits=0,
        projection_dim=64,
        value_spec_key="hadamard_lm_k4v4",
        compute_interpretation="aggressive pure Hadamard-LM key reconstruction",
    ),
    "hadamard_lm_k2_qjl": QJLSpec(
        name="hadamard_lm_k2_qjl",
        base_k_bits=2,
        residual_bits=1,
        projection_dim=64,
        value_spec_key="hadamard_lm_k4v4",
        compute_interpretation="Hadamard-LM key plus QJL residual correction",
    ),
    "hadamard_lm_k3_qjl": QJLSpec(
        name="hadamard_lm_k3_qjl",
        base_k_bits=3,
        residual_bits=1,
        projection_dim=64,
        value_spec_key="hadamard_lm_k4v4",
        compute_interpretation="stronger Hadamard-LM key plus QJL residual correction",
    ),
}


STAGE_C_STRUCTURED_ATTENTION_SPECS: dict[str, StageCStructuredAttentionSpec] = {
    "attn_identity_fp16": StageCStructuredAttentionSpec(
        name="attn_identity_fp16",
        linear_spec=None,
        kv_spec_key="fp16",
        value_path="reconstruct",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="identity attention wrapper; no quantization",
    ),
    "attn_kv_hlm_k4v4_reconstruct": StageCStructuredAttentionSpec(
        name="attn_kv_hlm_k4v4_reconstruct",
        linear_spec=None,
        kv_spec_key="hadamard_lm_k4v4",
        value_path="reconstruct",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="KV-only HLM K4V4 with value reconstruction",
    ),
    "attn_kv_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_kv_hlm_k4v4",
        linear_spec=None,
        kv_spec_key="hadamard_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="KV-only HLM K4V4 with value rotation absorbed into o_proj",
    ),
    "attn_kv_hlm_k4v4_h32": StageCStructuredAttentionSpec(
        name="attn_kv_hlm_k4v4_h32",
        linear_spec=None,
        kv_spec_key="hadamard_lm_k4v4_h32",
        value_path="o_proj_absorb",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="KV-only HLM K4V4 with head-internal H32 value rotation absorbed into o_proj",
    ),
    "attn_kv_hlm_k3v4": StageCStructuredAttentionSpec(
        name="attn_kv_hlm_k3v4",
        linear_spec=None,
        kv_spec_key="hadamard_lm_k3v4",
        value_path="o_proj_absorb",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="KV-only HLM K3V4 with value rotation absorbed into o_proj",
    ),
    "attn_kv_hlm_k4v3": StageCStructuredAttentionSpec(
        name="attn_kv_hlm_k4v3",
        linear_spec=None,
        kv_spec_key="hadamard_lm_k4v3",
        value_path="o_proj_absorb",
        quantize_qkv=False,
        quantize_o=False,
        compute_interpretation="KV-only HLM K4V3 with value rotation absorbed into o_proj",
    ),
    "attn_rot_lm_w4a4_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_rot_lm_w4a4_hlm_k4v4",
        linear_spec=STAGE_B_LINEAR_SPECS["rot_lm_w4a4"],
        kv_spec_key="hadamard_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="rotated LM W4A4 plus HLM K4V4 with value rotation absorbed into o_proj",
    ),
    "attn_mxfp4_w4a4_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_mxfp4_w4a4_hlm_k4v4",
        linear_spec=STAGE_B_LINEAR_SPECS["mxfp4_w4a4"],
        kv_spec_key="hadamard_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="MXFP4 W4A4 linear fake quant plus HLM K4V4",
    ),
    "attn_rot_mxfp4_w4a4_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_rot_mxfp4_w4a4_hlm_k4v4",
        linear_spec=STAGE_B_LINEAR_SPECS["rot_mxfp4_w4a4"],
        kv_spec_key="hadamard_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="Hadamard-rotated MXFP4 W4A4 linear fake quant plus HLM K4V4",
    ),
    "attn_rot_lm_w3a4_hlm_k3v4": StageCStructuredAttentionSpec(
        name="attn_rot_lm_w3a4_hlm_k3v4",
        linear_spec=STAGE_B_LINEAR_SPECS["rot_lm_w3a4"],
        kv_spec_key="hadamard_lm_k3v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="rotated LM W3A4 plus HLM K3V4 with value rotation absorbed into o_proj",
    ),
    "attn_rot_lm_w4a3_hlm_k4v3": StageCStructuredAttentionSpec(
        name="attn_rot_lm_w4a3_hlm_k4v3",
        linear_spec=STAGE_B_LINEAR_SPECS["rot_lm_w4a3"],
        kv_spec_key="hadamard_lm_k4v3",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="rotated LM W4A3 plus HLM K4V3 with value rotation absorbed into o_proj",
    ),
    "attn_randhadamard_lm_w4a4_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_randhadamard_lm_w4a4_hlm_k4v4",
        linear_spec=STAGE_B_LINEAR_SPECS["randhadamard_lm_w4a4"],
        kv_spec_key="randhadamard_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="randomized Hadamard LM W4A4 plus randomized HLM K4V4",
    ),
    "attn_randhadamard_lm_w3a4_hlm_k3v4": StageCStructuredAttentionSpec(
        name="attn_randhadamard_lm_w3a4_hlm_k3v4",
        linear_spec=STAGE_B_LINEAR_SPECS["randhadamard_lm_w3a4"],
        kv_spec_key="randhadamard_lm_k3v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="randomized Hadamard LM W3A4 plus randomized HLM K3V4",
    ),
    "attn_randortho_lm_w4a4_hlm_k4v4": StageCStructuredAttentionSpec(
        name="attn_randortho_lm_w4a4_hlm_k4v4",
        linear_spec=STAGE_B_LINEAR_SPECS["randortho_lm_w4a4"],
        kv_spec_key="randortho_lm_k4v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="dense random orthogonal LM W4A4 plus random-orthogonal K4V4",
    ),
    "attn_randortho_lm_w3a4_hlm_k3v4": StageCStructuredAttentionSpec(
        name="attn_randortho_lm_w3a4_hlm_k3v4",
        linear_spec=STAGE_B_LINEAR_SPECS["randortho_lm_w3a4"],
        kv_spec_key="randortho_lm_k3v4",
        value_path="o_proj_absorb",
        quantize_qkv=True,
        quantize_o=True,
        compute_interpretation="dense random orthogonal LM W3A4 plus random-orthogonal K3V4",
    ),
}

def make_head_signs(
    head_dim: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a seeded random sign diagonal for optional randomized Hadamard."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (head_dim,), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(device=device, dtype=dtype)
    return signs


def headwise_hadamard(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Apply H or H D along the last dimension of attention heads."""
    values = x
    if signs is not None:
        values = values * signs.to(device=x.device, dtype=x.dtype)
    return fwht(values, dim=-1, normalize=True)


def inverse_headwise_hadamard(x: torch.Tensor, signs: torch.Tensor | None = None) -> torch.Tensor:
    """Invert headwise_hadamard; H is self-inverse and D^{-1}=D."""
    values = fwht(x, dim=-1, normalize=True)
    if signs is not None:
        values = values * signs.to(device=x.device, dtype=x.dtype)
    return values


def make_head_rotation_matrix(
    head_dim: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a deterministic dense orthogonal matrix for head-wise ablation."""
    return random_orthogonal_matrix(head_dim, seed=seed, device=device, dtype=dtype)


def headwise_rotation(
    x: torch.Tensor,
    *,
    rotation_backend: str = "hadamard",
    signs: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
    block_size: int | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply the selected head-wise orthonormal rotation along head_dim."""
    head_dim = x.shape[-1]
    effective_block_size = block_size or head_dim
    if rotation_backend == "hadamard":
        if effective_block_size != head_dim:
            return block_rotation_last_dim(
                x,
                block_size=effective_block_size,
                rotation_backend="hadamard",
                inverse=inverse,
            )
        return headwise_hadamard(x, signs=None)
    if rotation_backend == "randomized_hadamard":
        if effective_block_size != head_dim:
            return block_rotation_last_dim(
                x,
                block_size=effective_block_size,
                rotation_backend="randomized_hadamard",
                inverse=inverse,
            )
        return inverse_headwise_hadamard(x, signs=signs) if inverse else headwise_hadamard(x, signs=signs)
    if rotation_backend == "random_orthogonal":
        if effective_block_size != head_dim:
            raise ValueError("random_orthogonal KV rotation only supports full-head blocks.")
        if matrix is None:
            matrix = make_head_rotation_matrix(
                head_dim,
                seed=0,
                device=x.device,
                dtype=x.dtype,
            )
        matrix = matrix.to(device=x.device, dtype=x.dtype)
        return x @ (matrix.t() if inverse else matrix)
    raise ValueError(f"Unsupported head-wise rotation backend: {rotation_backend}")


def per_head_rms(x: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return (x.float() / scale).to(dtype=x.dtype), scale.to(dtype=x.dtype)


def _lloyd_max_centroid_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    codebook = torch.tensor(gaussian_lloyd_max_codebook(bits), device=x.device, dtype=torch.float32)
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    # The codebook is tiny. This avoids torch.bucketize paths that are slow on MPS.
    indices = torch.zeros_like(x.float(), dtype=torch.long)
    for boundary in boundaries:
        indices = indices + (x.float() > boundary).to(torch.long)
    return codebook[indices].to(dtype=x.dtype)


def absmax_quantize_per_head(x: torch.Tensor, bits: int, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric uniform fake quant with one scale per token/head vector."""
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(eps) / qmax
    q = torch.round(x.float() / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=x.dtype)


def hadamard_lm_quantize_per_head(
    x: torch.Tensor,
    bits: int,
    signs: torch.Tensor | None = None,
    rotation_backend: str = "hadamard",
    matrix: torch.Tensor | None = None,
    kv_block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reconstructed x, quantized rotated normalized x, and RMS scale."""
    normalized, scale = per_head_rms(x)
    rotated = headwise_rotation(
        normalized,
        rotation_backend=rotation_backend,
        signs=signs,
        matrix=matrix,
        block_size=kv_block_size,
    )
    quantized_rotated = _lloyd_max_centroid_quantize(rotated, bits)
    restored = headwise_rotation(
        quantized_rotated,
        rotation_backend=rotation_backend,
        signs=signs,
        matrix=matrix,
        block_size=kv_block_size,
        inverse=True,
    ) * scale
    return restored.to(dtype=x.dtype), quantized_rotated.to(dtype=x.dtype), scale.to(dtype=x.dtype)


def repeat_kv_heads(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def raw_attention_inner_product(
    query: torch.Tensor,
    key: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    key_states = repeat_kv_heads(key, num_key_value_groups)
    return torch.matmul(query.float(), key_states.float().transpose(2, 3))


def apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        query_len = scores.shape[-2]
        key_len = scores.shape[-1]
        query_positions = torch.arange(query_len, device=scores.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=scores.device).unsqueeze(0)
        past_len = max(key_len - query_len, 0)
        valid = key_positions <= (query_positions + past_len)
        causal_mask = torch.zeros((query_len, key_len), device=scores.device, dtype=scores.dtype)
        causal_mask = causal_mask.masked_fill(~valid, torch.finfo(scores.dtype).min)
        return scores + causal_mask
    return scores + attention_mask.to(device=scores.device, dtype=scores.dtype)


def attention_probs(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    masked = apply_attention_mask(scores, attention_mask)
    return F.softmax(masked, dim=-1, dtype=torch.float32).to(dtype=scores.dtype)


def attention_output_from_probs(
    probs: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    value_states = repeat_kv_heads(value, num_key_value_groups)
    return torch.matmul(probs, value_states.float()).to(dtype=value.dtype)


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> AttentionComputation:
    raw_ip = raw_attention_inner_product(query, key, num_key_value_groups)
    scores = raw_ip * scaling
    probs = attention_probs(scores, attention_mask)
    output_heads = attention_output_from_probs(probs, value, num_key_value_groups)
    return AttentionComputation(
        raw_inner_product=raw_ip,
        scores=scores,
        probs=probs,
        output_heads=output_heads,
        key_hat=key,
        value_hat=value,
        metadata={
            "method": "fp16",
            "bits": "K16V16",
            "k_bits": 16,
            "v_bits": 16,
            "kv_rotation": "none",
            "kv_quantizer": "none",
            "key_scale_granularity": "none",
            "value_scale_granularity": "none",
            "compute_interpretation": "baseline",
        },
    )


def _score_valid_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = attention_mask.to(device=scores.device)
    valid = mask >= 0
    return valid.expand_as(scores)


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return values[valid].float().mean() if valid.any() else values.float().mean()


def _masked_var(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return values[valid].float().var(unbiased=False) if valid.any() else values.float().var(unbiased=False)


def _masked_relative_mse(reference: torch.Tensor, candidate: torch.Tensor, valid: torch.Tensor) -> float:
    ref = reference.float()[valid]
    cand = candidate.float()[valid]
    if ref.numel() == 0:
        return relative_mse(reference, candidate)
    return float((ref - cand).square().mean().div(ref.square().mean().clamp_min(1e-12)).cpu())


def _masked_mse(reference: torch.Tensor, candidate: torch.Tensor, valid: torch.Tensor) -> float:
    ref = reference.float()[valid]
    cand = candidate.float()[valid]
    if ref.numel() == 0:
        return float((reference.float() - candidate.float()).square().mean().cpu())
    return float((ref - cand).square().mean().cpu())


def softmax_kl(reference_probs: torch.Tensor, candidate_probs: torch.Tensor, eps: float = 1e-12) -> float:
    ref = reference_probs.float().clamp_min(eps)
    cand = candidate_probs.float().clamp_min(eps)
    terms = torch.where(reference_probs.float() > eps, ref * (torch.log(ref) - torch.log(cand)), torch.zeros_like(ref))
    return float(terms.sum(dim=-1).mean().cpu())


def topk_attention_overlap(reference_probs: torch.Tensor, candidate_probs: torch.Tensor, k: int = 10) -> float:
    k = min(k, reference_probs.shape[-1])
    ref_idx = torch.topk(reference_probs.float(), k=k, dim=-1).indices
    cand_idx = torch.topk(candidate_probs.float(), k=k, dim=-1).indices
    matches = (ref_idx.unsqueeze(-1) == cand_idx.unsqueeze(-2)).any(dim=-1).float().mean()
    return float(matches.cpu())


def attention_quality_metrics(
    reference: AttentionComputation,
    candidate: AttentionComputation,
    attention_mask: torch.Tensor | None,
) -> dict[str, float]:
    valid = _score_valid_mask(reference.scores, attention_mask)
    ip_diff = candidate.raw_inner_product.float() - reference.raw_inner_product.float()
    metrics = {
        "ip_bias": float(_masked_mean(ip_diff, valid).cpu()),
        "ip_variance": float(_masked_var(ip_diff, valid).cpu()),
        "ip_relative_mse": _masked_relative_mse(reference.raw_inner_product, candidate.raw_inner_product, valid),
        "score_mse": _masked_mse(reference.scores, candidate.scores, valid),
        "score_relative_mse": _masked_relative_mse(reference.scores, candidate.scores, valid),
        "softmax_kl": softmax_kl(reference.probs, candidate.probs),
        "topk_overlap": topk_attention_overlap(reference.probs, candidate.probs),
        "output_relative_mse": relative_mse(reference.output_heads, candidate.output_heads),
        "output_cosine": cosine_similarity(reference.output_heads, candidate.output_heads),
        "key_relative_mse": relative_mse(reference.key_hat, candidate.key_hat),
        "value_relative_mse": relative_mse(reference.value_hat, candidate.value_hat),
    }
    return metrics


def quantize_value_for_kv(
    value: torch.Tensor,
    spec: KVQuantSpec,
    signs: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if spec.method == "fp16":
        return value, {"value_scale_granularity": "none"}
    if spec.method == "absmax":
        return absmax_quantize_per_head(value, spec.v_bits), {"value_scale_granularity": "per_token_head_absmax"}
    if spec.method == "hadamard_lm":
        value_hat, _value_rotated, _value_scale = hadamard_lm_quantize_per_head(
            value,
            spec.v_bits,
            signs=signs,
            rotation_backend=spec.rotation_backend,
            matrix=matrix,
            kv_block_size=spec.kv_block_size,
        )
        return value_hat, {"value_scale_granularity": "per_token_head_rms", "kv_block_size": spec.kv_block_size}
    raise ValueError(f"Unsupported KV method: {spec.method}")


def quantized_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    spec: KVQuantSpec,
    signs: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
) -> AttentionComputation:
    if spec.method == "fp16":
        return reference_attention(query, key, value, attention_mask, scaling, num_key_value_groups)

    if spec.method == "absmax":
        key_hat = absmax_quantize_per_head(key, spec.k_bits)
        value_hat, value_meta = quantize_value_for_kv(value, spec, signs=signs, matrix=matrix)
        raw_ip = raw_attention_inner_product(query, key_hat, num_key_value_groups)
    elif spec.method == "hadamard_lm":
        key_hat, key_rotated_hat, key_scale = hadamard_lm_quantize_per_head(
            key,
            spec.k_bits,
            signs=signs,
            rotation_backend=spec.rotation_backend,
            matrix=matrix,
            kv_block_size=spec.kv_block_size,
        )
        query_rotated = headwise_rotation(
            query,
            rotation_backend=spec.rotation_backend,
            signs=signs,
            matrix=matrix,
            block_size=spec.kv_block_size,
        )
        key_rotated_actual = key_rotated_hat * key_scale
        raw_ip = raw_attention_inner_product(query_rotated, key_rotated_actual, num_key_value_groups)
        value_hat, value_meta = quantize_value_for_kv(value, spec, signs=signs, matrix=matrix)
    else:
        raise ValueError(f"Unsupported KV method: {spec.method}")

    scores = raw_ip * scaling
    probs = attention_probs(scores, attention_mask)
    output_heads = attention_output_from_probs(probs, value_hat, num_key_value_groups)
    metadata = {
        "method": spec.method,
        "bits": spec.label,
        "k_bits": spec.k_bits,
        "v_bits": spec.v_bits,
        "kv_rotation": spec.rotation,
        "rotation_backend": spec.rotation_backend,
        "kv_block_size": spec.kv_block_size,
        "kv_quantizer": spec.quantizer,
        "key_scale_granularity": (
            f"per_token_head_h{spec.kv_block_size}_rms" if spec.method == "hadamard_lm" else "per_token_head_absmax"
        ),
        "compute_interpretation": spec.compute_interpretation,
        **value_meta,
    }
    return AttentionComputation(
        raw_inner_product=raw_ip,
        scores=scores,
        probs=probs,
        output_heads=output_heads,
        key_hat=key_hat,
        value_hat=value_hat,
        metadata=metadata,
    )


def quantized_kv_attention_o_proj_absorb(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    spec: KVQuantSpec,
    signs: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
) -> AttentionComputation:
    """Attention where value remains in head-wise rotated domain for o_proj absorption.

    Q/K score still uses independent H64 rotations per head. Value output heads
    are returned in the same rotated H64 domain, so the caller must use
    W_o H64_blockdiag instead of the original o_proj weight.
    """
    rotation_backend = spec.rotation_backend if spec.rotation_backend != "none" else "hadamard"
    if spec.method == "fp16":
        query_rotated = headwise_rotation(query, rotation_backend=rotation_backend, signs=signs, matrix=matrix, block_size=spec.kv_block_size)
        key_rotated = headwise_rotation(key, rotation_backend=rotation_backend, signs=signs, matrix=matrix, block_size=spec.kv_block_size)
        raw_ip = raw_attention_inner_product(query_rotated, key_rotated, num_key_value_groups)
        value_rotated = headwise_rotation(value, rotation_backend=rotation_backend, signs=signs, matrix=matrix, block_size=spec.kv_block_size)
        value_hat = value
        value_meta = {"value_scale_granularity": "none"}
    elif spec.method == "hadamard_lm":
        key_hat, key_rotated_hat, key_scale = hadamard_lm_quantize_per_head(
            key,
            spec.k_bits,
            signs=signs,
            rotation_backend=spec.rotation_backend,
            matrix=matrix,
            kv_block_size=spec.kv_block_size,
        )
        query_rotated = headwise_rotation(
            query,
            rotation_backend=spec.rotation_backend,
            signs=signs,
            matrix=matrix,
            block_size=spec.kv_block_size,
        )
        key_rotated_actual = key_rotated_hat * key_scale
        raw_ip = raw_attention_inner_product(query_rotated, key_rotated_actual, num_key_value_groups)
        value_hat, value_rotated_hat, value_scale = hadamard_lm_quantize_per_head(
            value,
            spec.v_bits,
            signs=signs,
            rotation_backend=spec.rotation_backend,
            matrix=matrix,
            kv_block_size=spec.kv_block_size,
        )
        value_rotated = value_rotated_hat * value_scale
        value_meta = {"value_scale_granularity": f"per_token_head_h{spec.kv_block_size}_rms"}
    else:
        raise ValueError(f"o_proj_absorb currently supports fp16 and hadamard_lm KV methods, got {spec.method}.")

    scores = raw_ip * scaling
    probs = attention_probs(scores, attention_mask)
    output_heads = attention_output_from_probs(probs, value_rotated, num_key_value_groups)
    return AttentionComputation(
        raw_inner_product=raw_ip,
        scores=scores,
        probs=probs,
        output_heads=output_heads,
        key_hat=key if spec.method == "fp16" else key_hat,
        value_hat=value_hat,
        metadata={
            "method": spec.method,
            "bits": spec.label,
            "k_bits": spec.k_bits,
            "v_bits": spec.v_bits,
            "kv_rotation": spec.rotation,
            "rotation_backend": spec.rotation_backend,
            "kv_block_size": spec.kv_block_size,
            "kv_quantizer": spec.quantizer,
            "key_scale_granularity": "none" if spec.method == "fp16" else f"per_token_head_h{spec.kv_block_size}_rms",
            "value_path": "o_proj_absorb",
            "compute_interpretation": spec.compute_interpretation,
            **value_meta,
        },
    )


def evaluate_kv_quantization(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    spec: KVQuantSpec,
    signs: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
) -> dict[str, object]:
    reference = reference_attention(query, key, value, attention_mask, scaling, num_key_value_groups)
    candidate = quantized_kv_attention(
        query,
        key,
        value,
        attention_mask,
        scaling,
        num_key_value_groups,
        spec,
        signs=signs,
        matrix=matrix,
    )
    return {**candidate.metadata, **attention_quality_metrics(reference, candidate, attention_mask)}


def make_qjl_matrix(
    projection_dim: int,
    head_dim: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn((projection_dim, head_dim), generator=generator, dtype=torch.float32)
    return matrix.to(device=device, dtype=dtype)


def _qjl_score_correction(
    query: torch.Tensor,
    residual: torch.Tensor,
    projection: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    residual_norm = residual.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    query_proj = F.linear(query.float(), projection.float())
    residual_proj_sign = torch.sign(F.linear(residual.float(), projection.float()))
    residual_proj_sign = torch.where(residual_proj_sign == 0, torch.ones_like(residual_proj_sign), residual_proj_sign)
    residual_proj_sign = repeat_kv_heads(residual_proj_sign, num_key_value_groups)
    residual_norm = repeat_kv_heads(residual_norm, num_key_value_groups)
    correction = torch.matmul(query_proj, residual_proj_sign.transpose(2, 3))
    correction = correction * (math.sqrt(math.pi / 2.0) / projection.shape[0])
    return correction * residual_norm.transpose(2, 3)


def qjl_residual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    qjl_spec: QJLSpec,
    *,
    seed: int,
    signs: torch.Tensor | None = None,
) -> AttentionComputation:
    key_base, _key_rotated, _key_scale = hadamard_lm_quantize_per_head(
        key,
        qjl_spec.base_k_bits,
        signs=signs,
    )
    residual = key.float() - key_base.float()
    raw_ip = raw_attention_inner_product(query, key_base, num_key_value_groups)
    if qjl_spec.residual_bits:
        projection = make_qjl_matrix(
            qjl_spec.projection_dim,
            query.shape[-1],
            seed=seed,
            device=query.device,
            dtype=torch.float32,
        )
        raw_ip = raw_ip + _qjl_score_correction(query, residual, projection, num_key_value_groups)

    value_spec = STAGE_C_KV_SPECS[qjl_spec.value_spec_key]
    value_hat, value_meta = quantize_value_for_kv(value, value_spec, signs=signs)
    scores = raw_ip * scaling
    probs = attention_probs(scores, attention_mask)
    output_heads = attention_output_from_probs(probs, value_hat, num_key_value_groups)
    metadata = {
        "method": qjl_spec.name,
        "bits": qjl_spec.label,
        "k_bits": qjl_spec.base_k_bits,
        "base_k_bits": qjl_spec.base_k_bits,
        "residual_bits": qjl_spec.residual_bits,
        "projection_dim": qjl_spec.projection_dim,
        "v_bits": value_spec.v_bits,
        "kv_rotation": "headwise_hadamard",
        "kv_quantizer": "gaussian_lloyd_max+qjl" if qjl_spec.residual_bits else "gaussian_lloyd_max",
        "key_scale_granularity": "per_token_head_rms",
        "compute_interpretation": qjl_spec.compute_interpretation,
        **value_meta,
    }
    return AttentionComputation(
        raw_inner_product=raw_ip,
        scores=scores,
        probs=probs,
        output_heads=output_heads,
        key_hat=key_base,
        value_hat=value_hat,
        metadata=metadata,
    )


def evaluate_qjl_residual(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    qjl_spec: QJLSpec,
    *,
    seed: int,
    signs: torch.Tensor | None = None,
) -> dict[str, object]:
    reference = reference_attention(query, key, value, attention_mask, scaling, num_key_value_groups)
    candidate = qjl_residual_attention(
        query,
        key,
        value,
        attention_mask,
        scaling,
        num_key_value_groups,
        qjl_spec,
        seed=seed,
        signs=signs,
    )
    return {**candidate.metadata, **attention_quality_metrics(reference, candidate, attention_mask)}


def invariance_metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    signs: torch.Tensor | None = None,
) -> dict[str, float]:
    reference = reference_attention(query, key, value, attention_mask, scaling, num_key_value_groups)
    query_rotated = headwise_hadamard(query, signs=signs)
    key_rotated = headwise_hadamard(key, signs=signs)
    raw_ip = raw_attention_inner_product(query_rotated, key_rotated, num_key_value_groups)
    scores = raw_ip * scaling
    probs = attention_probs(scores, attention_mask)
    output_heads = attention_output_from_probs(probs, value, num_key_value_groups)
    candidate = AttentionComputation(
        raw_inner_product=raw_ip,
        scores=scores,
        probs=probs,
        output_heads=output_heads,
        key_hat=key,
        value_hat=value,
        metadata={},
    )
    valid = _score_valid_mask(reference.scores, attention_mask)
    return {
        "score_relative_mse": _masked_relative_mse(reference.scores, candidate.scores, valid),
        "max_score_abs_diff": float((reference.scores.float()[valid] - candidate.scores.float()[valid]).abs().max().cpu()),
        "softmax_kl": softmax_kl(reference.probs, candidate.probs),
        "output_relative_mse": relative_mse(reference.output_heads, candidate.output_heads),
        "output_cosine": cosine_similarity(reference.output_heads, candidate.output_heads),
    }


def projection_error_metrics(
    reference_q: torch.Tensor,
    reference_k: torch.Tensor,
    reference_v: torch.Tensor,
    candidate_q: torch.Tensor,
    candidate_k: torch.Tensor,
    candidate_v: torch.Tensor,
) -> dict[str, float]:
    q_metrics = tensor_metrics(reference_q, candidate_q)
    k_metrics = tensor_metrics(reference_k, candidate_k)
    v_metrics = tensor_metrics(reference_v, candidate_v)
    return {
        "q_proj_relative_mse": q_metrics["relative_mse"],
        "k_proj_relative_mse": k_metrics["relative_mse"],
        "v_proj_relative_mse": v_metrics["relative_mse"],
        "projection_relative_mse": (q_metrics["relative_mse"] + k_metrics["relative_mse"] + v_metrics["relative_mse"])
        / 3.0,
        "projection_cosine": (q_metrics["cosine"] + k_metrics["cosine"] + v_metrics["cosine"]) / 3.0,
    }
