from __future__ import annotations

import torch
import torch.nn.functional as F

from rotationquant.metrics import relative_mse
from rotationquant.modeling import iter_llama_decoder_layers
from rotationquant.stage_b import (
    STAGE_B_METHODS,
    prepare_stage_b_weight,
    quantize_stage_b_domain,
    quantize_stage_b_activation_domain,
)
from rotationquant.stage_c import (
    STAGE_C_KV_SPECS,
    STAGE_C_STRUCTURED_ATTENTION_SPECS,
    StageCStructuredAttentionSpec,
    headwise_rotation,
    make_head_rotation_matrix,
    make_head_signs,
    projection_error_metrics,
    quantized_kv_attention,
    quantized_kv_attention_o_proj_absorb,
)
from rotationquant.rotations import block_rotation_last_dim

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - import is validated by smoke scripts.
    raise RuntimeError("Stage C attention wrappers require Hugging Face LLaMA modules.") from exc


def apply_stage_c_attention_fake_quant_(
    model: torch.nn.Module,
    method_name: str,
    *,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> list[dict[str, object]]:
    """Replace self-attention modules with the canonical Stage C wrapper."""
    return apply_stage_c_structured_attention_(
        model,
        method_name,
        block_size=block_size,
        mxfp4_group_size=mxfp4_group_size,
        rotation_seed=rotation_seed,
        use_random_signs=use_random_signs,
        sign_seed=sign_seed,
    )


def fake_quant_attention_from_record(
    record,
    *,
    method_name: str,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one captured attention input through the canonical Stage C wrapper."""
    return fake_quant_structured_attention_from_record(
        record,
        method_name=method_name,
        block_size=block_size,
        mxfp4_group_size=mxfp4_group_size,
        rotation_seed=rotation_seed,
        use_random_signs=use_random_signs,
        sign_seed=sign_seed,
    )


def attention_layer_metrics_from_details(record, candidate_output: torch.Tensor, details: dict[str, torch.Tensor]) -> dict[str, float]:
    projection_metrics = projection_error_metrics(
        record.q_proj_out,
        record.k_proj_out,
        record.v_proj_out,
        details["q_proj_out"],
        details["k_proj_out"],
        details["v_proj_out"],
    )
    return {
        **projection_metrics,
        "layer_output_relative_mse": relative_mse(record.output, candidate_output),
        "layer_output_cosine": float(
            torch.nn.functional.cosine_similarity(
                record.output.float().reshape(1, -1),
                candidate_output.float().reshape(1, -1),
            ).cpu()
        ),
    }


def absorb_o_proj_weight_headwise(
    weight: torch.Tensor,
    *,
    head_dim: int,
    rotation_block_size: int | None = None,
    signs: torch.Tensor | None = None,
    rotation_backend: str = "hadamard",
    rotation_seed: int = 0,
) -> torch.Tensor:
    """Return W_o R where R is the independent per-head rotation.

    The weight shape is [out_features, hidden_size]. The input dimension is
    interpreted as concatenated attention heads; no cross-head mixing is used.
    """
    return block_rotation_last_dim(
        weight,
        block_size=rotation_block_size or head_dim,
        rotation_backend=rotation_backend,
        seed=rotation_seed,
    )


class StageCStructuredAttentionWrapper(torch.nn.Module):
    """Canonical Stage C attention wrapper with optional value rotation absorption."""

    def __init__(
        self,
        attention_module: torch.nn.Module,
        *,
        method_name: str,
        block_size: int = 128,
        mxfp4_group_size: int = 32,
        rotation_seed: int = 0,
        use_random_signs: bool = False,
        sign_seed: int = 0,
        record_details: bool = False,
    ) -> None:
        super().__init__()
        if method_name not in STAGE_C_STRUCTURED_ATTENTION_SPECS:
            raise ValueError(f"Unknown Stage C attention method: {method_name}")
        self.method_name = method_name
        self.spec: StageCStructuredAttentionSpec = STAGE_C_STRUCTURED_ATTENTION_SPECS[method_name]
        self.block_size = block_size
        self.mxfp4_group_size = mxfp4_group_size
        self.rotation_seed = rotation_seed
        self.record_details = record_details
        self.config = attention_module.config
        self.layer_idx = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.num_key_value_groups = attention_module.num_key_value_groups
        self.scaling = float(attention_module.scaling)
        self.attention_dropout = getattr(attention_module, "attention_dropout", 0.0)
        self.last_details: dict[str, torch.Tensor] | None = None

        kv_spec = STAGE_C_KV_SPECS[self.spec.kv_spec_key]
        self.kv_block_size = kv_spec.kv_block_size
        self.head_rotation_backend = "randomized_hadamard" if use_random_signs else kv_spec.rotation_backend
        self.head_rotation_seed = sign_seed if use_random_signs else rotation_seed
        base_dtype = attention_module.q_proj.weight.dtype
        signs = (
            make_head_signs(
                self.head_dim,
                seed=self.head_rotation_seed,
                device="cpu",
                dtype=base_dtype,
            )
            if self.head_rotation_backend == "randomized_hadamard"
            else None
        )
        rotation_matrix = (
            make_head_rotation_matrix(
                self.head_dim,
                seed=self.head_rotation_seed,
                device="cpu",
                dtype=base_dtype,
            )
            if self.head_rotation_backend == "random_orthogonal"
            else None
        )
        self.register_buffer("head_signs", signs)
        self.register_buffer("head_rotation_matrix", rotation_matrix)

        if self.spec.quantize_qkv:
            assert self.spec.linear_spec is not None
            q_weight, _ = prepare_stage_b_weight(
                attention_module.q_proj.weight.detach().cpu(),
                self.spec.linear_spec,
                block_size,
                mxfp4_group_size=mxfp4_group_size,
                rotation_seed=rotation_seed,
            )
            k_weight, _ = prepare_stage_b_weight(
                attention_module.k_proj.weight.detach().cpu(),
                self.spec.linear_spec,
                block_size,
                mxfp4_group_size=mxfp4_group_size,
                rotation_seed=rotation_seed,
            )
            v_weight, _ = prepare_stage_b_weight(
                attention_module.v_proj.weight.detach().cpu(),
                self.spec.linear_spec,
                block_size,
                mxfp4_group_size=mxfp4_group_size,
                rotation_seed=rotation_seed,
            )
        else:
            q_weight = attention_module.q_proj.weight.detach().cpu()
            k_weight = attention_module.k_proj.weight.detach().cpu()
            v_weight = attention_module.v_proj.weight.detach().cpu()

        o_weight = attention_module.o_proj.weight.detach().cpu()
        if self.spec.value_path == "o_proj_absorb":
            o_weight = absorb_o_proj_weight_headwise(
                o_weight,
                head_dim=self.head_dim,
                rotation_block_size=self.kv_block_size,
                signs=signs,
                rotation_backend=self.head_rotation_backend,
                rotation_seed=self.head_rotation_seed,
            )
            if self.spec.quantize_o:
                assert self.spec.linear_spec is not None
                method = STAGE_B_METHODS[self.spec.linear_spec.method]
                o_weight, _ = quantize_stage_b_domain(
                    o_weight,
                    self.spec.linear_spec.w_bits,
                    method.quantizer,
                    block_size=self.kv_block_size,
                    mxfp4_group_size=mxfp4_group_size,
                )
        elif self.spec.value_path == "reconstruct":
            if self.spec.quantize_o:
                assert self.spec.linear_spec is not None
                o_weight, _ = prepare_stage_b_weight(
                    o_weight,
                    self.spec.linear_spec,
                    block_size,
                    mxfp4_group_size=mxfp4_group_size,
                    rotation_seed=rotation_seed,
                )
        else:
            raise ValueError(f"Unsupported value_path: {self.spec.value_path}")

        self.register_buffer("q_weight", q_weight)
        self.register_buffer("k_weight", k_weight)
        self.register_buffer("v_weight", v_weight)
        self.register_buffer("o_weight", o_weight)
        self.register_buffer("q_bias", self._bias_or_none(attention_module.q_proj))
        self.register_buffer("k_bias", self._bias_or_none(attention_module.k_proj))
        self.register_buffer("v_bias", self._bias_or_none(attention_module.v_proj))
        self.register_buffer("o_bias", self._bias_or_none(attention_module.o_proj))

    @staticmethod
    def _bias_or_none(linear: torch.nn.Linear) -> torch.Tensor | None:
        bias = getattr(linear, "bias", None)
        return bias.detach().cpu() if bias is not None else None

    def _linear_dtype(self) -> torch.dtype:
        return self.q_weight.dtype

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(dtype=self._linear_dtype())
        if self.spec.quantize_qkv:
            assert self.spec.linear_spec is not None
            qkv_input, _ = quantize_stage_b_activation_domain(
                hidden_states,
                self.spec.linear_spec,
                block_size=self.block_size,
                mxfp4_group_size=self.mxfp4_group_size,
                rotation_seed=self.rotation_seed,
            )
        else:
            qkv_input = hidden_states
        return (
            F.linear(qkv_input, self.q_weight, self.q_bias),
            F.linear(qkv_input, self.k_weight, self.k_bias),
            F.linear(qkv_input, self.v_weight, self.v_bias),
        )

    def _project_o(self, attn_output: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        attn_output = attn_output.to(dtype=self._linear_dtype())
        if self.spec.quantize_o:
            assert self.spec.linear_spec is not None
            method = STAGE_B_METHODS[self.spec.linear_spec.method]
            if self.spec.value_path == "o_proj_absorb":
                o_input, _ = quantize_stage_b_domain(
                    attn_output,
                    self.spec.linear_spec.a_bits,
                    method.quantizer,
                    block_size=self.kv_block_size,
                    mxfp4_group_size=self.mxfp4_group_size,
                )
            else:
                o_input, _ = quantize_stage_b_activation_domain(
                    attn_output,
                    self.spec.linear_spec,
                    block_size=self.block_size,
                    mxfp4_group_size=self.mxfp4_group_size,
                    rotation_seed=self.rotation_seed,
                )
        else:
            o_input = attn_output
        return F.linear(o_input, self.o_weight, self.o_bias).to(dtype=hidden_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_proj, key_proj, value_proj = self._project_qkv(hidden_states)
        query_states = query_proj.view(hidden_shape).transpose(1, 2)
        key_states = key_proj.view(hidden_shape).transpose(1, 2)
        value_states = value_proj.view(hidden_shape).transpose(1, 2)
        query_before_rope = query_states
        key_before_rope = key_states
        value_before_quant = value_states

        if position_embeddings is None:
            raise ValueError("Stage C wrapper expects post-LlamaModel position_embeddings for RoPE.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        kv_spec = STAGE_C_KV_SPECS[self.spec.kv_spec_key]
        if self.spec.value_path == "o_proj_absorb":
            attention = quantized_kv_attention_o_proj_absorb(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                kv_spec,
                signs=self.head_signs,
                matrix=self.head_rotation_matrix,
            )
            pre_o_reference = headwise_rotation(
                attention.output_heads,
                rotation_backend=self.head_rotation_backend,
                signs=self.head_signs,
                matrix=self.head_rotation_matrix,
                block_size=self.kv_block_size,
                inverse=True,
            )
        else:
            attention = quantized_kv_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                kv_spec,
                signs=self.head_signs,
                matrix=self.head_rotation_matrix,
            )
            pre_o_reference = attention.output_heads

        attn_output = attention.output_heads.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = self._project_o(attn_output, hidden_states.dtype)

        if self.record_details:
            self.last_details = {
                "q_proj_out": query_before_rope.detach().cpu(),
                "k_proj_out": key_before_rope.detach().cpu(),
                "v_proj_out": value_before_quant.detach().cpu(),
                "q_rope": query_states.detach().cpu(),
                "k_rope": key_states.detach().cpu(),
                "raw_inner_product": attention.raw_inner_product.detach().cpu(),
                "scores": attention.scores.detach().cpu(),
                "attn_probs": attention.probs.detach().cpu(),
                "attn_output_heads": attention.output_heads.detach().cpu(),
                "attn_output_heads_reference_domain": pre_o_reference.detach().cpu(),
                "key_hat": attention.key_hat.detach().cpu(),
                "value_hat": attention.value_hat.detach().cpu(),
                "output": output.detach().cpu(),
            }
        return output, attention.probs


def apply_stage_c_structured_attention_(
    model: torch.nn.Module,
    method_name: str,
    *,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> list[dict[str, object]]:
    """Replace self-attention modules with canonical Stage C wrappers."""
    if method_name not in STAGE_C_STRUCTURED_ATTENTION_SPECS:
        raise ValueError(f"Unknown Stage C model method: {method_name}")
    spec = STAGE_C_STRUCTURED_ATTENTION_SPECS[method_name]
    records: list[dict[str, object]] = []
    for layer_index, layer in iter_llama_decoder_layers(model):
        original_attention = layer.self_attn
        layer.self_attn = StageCStructuredAttentionWrapper(
            original_attention,
            method_name=method_name,
            block_size=block_size,
            mxfp4_group_size=mxfp4_group_size,
            rotation_seed=rotation_seed + layer_index,
            use_random_signs=use_random_signs,
            sign_seed=sign_seed + layer_index,
        )
        kv_spec = STAGE_C_KV_SPECS[spec.kv_spec_key]
        method = STAGE_B_METHODS[spec.linear_spec.method].name if spec.linear_spec is not None else "fp16"
        records.append(
            {
                "layer": f"model.layers.{layer_index}.self_attn",
                "method_key": method_name,
                "linear_method": method,
                "linear_bits": spec.linear_spec.label if spec.linear_spec is not None else "FP16",
                "w_bits": spec.linear_spec.w_bits if spec.linear_spec is not None else 16,
                "a_bits": spec.linear_spec.a_bits if spec.linear_spec is not None else 16,
                "kv_method": kv_spec.method,
                "kv_bits": kv_spec.label,
                "k_bits": kv_spec.k_bits,
                "v_bits": kv_spec.v_bits,
                "kv_rotation_backend": kv_spec.rotation_backend,
                "kv_block_size": kv_spec.kv_block_size,
                "linear_rotation_backend": (
                    STAGE_B_METHODS[spec.linear_spec.method].rotation_backend or "none"
                    if spec.linear_spec is not None
                    else "none"
                ),
                "rotation_seed": rotation_seed + layer_index,
                "block_size": block_size,
                "o_proj_domain_block_size": kv_spec.kv_block_size if spec.value_path == "o_proj_absorb" else block_size,
                "mxfp4_group_size": mxfp4_group_size,
                "value_path": spec.value_path,
                "compute_interpretation": spec.compute_interpretation,
            }
        )
    return records


def fake_quant_structured_attention_from_record(
    record,
    *,
    method_name: str,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one captured attention input through the canonical Stage C wrapper."""
    wrapper = StageCStructuredAttentionWrapper(
        record.module,
        method_name=method_name,
        block_size=block_size,
        mxfp4_group_size=mxfp4_group_size,
        rotation_seed=rotation_seed + record.layer_index,
        use_random_signs=use_random_signs,
        sign_seed=sign_seed + record.layer_index,
        record_details=True,
    ).float()
    with torch.no_grad():
        output, _ = wrapper(
            record.input.float(),
            position_embeddings=tuple(t.float() for t in record.position_embeddings),
            attention_mask=record.attention_mask.float() if record.attention_mask is not None else None,
        )
    assert wrapper.last_details is not None
    return output.detach().cpu(), wrapper.last_details
