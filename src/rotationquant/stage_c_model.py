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
    STAGE_C_ATTENTION_SPECS,
    STAGE_C_KV_SPECS,
    STAGE_C_QJL_SPECS,
    STAGE_C_REFINE_ATTENTION_SPECS,
    StageCAttentionSpec,
    StageCRefineAttentionSpec,
    headwise_hadamard,
    inverse_headwise_hadamard,
    make_head_signs,
    projection_error_metrics,
    qjl_residual_attention,
    quantized_kv_attention,
    quantized_kv_attention_o_proj_absorb,
)

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - import is validated by smoke scripts.
    raise RuntimeError("Stage C attention wrappers require Hugging Face LLaMA modules.") from exc


class StageCAttentionWrapper(torch.nn.Module):
    """Attention-only fake quant wrapper for Stage C PPL and layer experiments."""

    def __init__(
        self,
        attention_module: torch.nn.Module,
        *,
        method_name: str,
        block_size: int = 128,
        use_random_signs: bool = False,
        sign_seed: int = 0,
        qjl_seed: int = 0,
        record_details: bool = False,
    ) -> None:
        super().__init__()
        if method_name not in STAGE_C_ATTENTION_SPECS:
            raise ValueError(f"Unknown Stage C attention method: {method_name}")
        self.method_name = method_name
        self.spec: StageCAttentionSpec = STAGE_C_ATTENTION_SPECS[method_name]
        if self.spec.linear_spec is None:
            raise ValueError("StageCAttentionWrapper is only used for non-FP16 methods.")
        self.block_size = block_size
        self.record_details = record_details
        self.qjl_seed = qjl_seed
        self.config = attention_module.config
        self.layer_idx = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.num_key_value_groups = attention_module.num_key_value_groups
        self.scaling = float(attention_module.scaling)
        self.attention_dropout = getattr(attention_module, "attention_dropout", 0.0)
        self.last_details: dict[str, torch.Tensor] | None = None

        # Pre-quantize weights in the Stage B W/A domain. Construction is done on
        # CPU; model.to("mps") later moves these buffers for PPL runs.
        q_weight, _ = prepare_stage_b_weight(attention_module.q_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
        k_weight, _ = prepare_stage_b_weight(attention_module.k_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
        v_weight, _ = prepare_stage_b_weight(attention_module.v_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
        o_weight, _ = prepare_stage_b_weight(attention_module.o_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("k_weight", k_weight)
        self.register_buffer("v_weight", v_weight)
        self.register_buffer("o_weight", o_weight)
        self.register_buffer("q_bias", self._bias_or_none(attention_module.q_proj))
        self.register_buffer("k_bias", self._bias_or_none(attention_module.k_proj))
        self.register_buffer("v_bias", self._bias_or_none(attention_module.v_proj))
        self.register_buffer("o_bias", self._bias_or_none(attention_module.o_proj))
        signs = (
            make_head_signs(
                self.head_dim,
                seed=sign_seed,
                device="cpu",
                dtype=q_weight.dtype,
            )
            if use_random_signs
            else None
        )
        self.register_buffer("head_signs", signs)

    @staticmethod
    def _bias_or_none(linear: torch.nn.Linear) -> torch.Tensor | None:
        bias = getattr(linear, "bias", None)
        return bias.detach().cpu() if bias is not None else None

    def _linear_dtype(self) -> torch.dtype:
        return self.q_weight.dtype

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(dtype=self._linear_dtype())
        qkv_input, _ = quantize_stage_b_activation_domain(
            hidden_states,
            self.spec.linear_spec,
            block_size=self.block_size,
        )
        query = F.linear(qkv_input, self.q_weight, self.q_bias)
        key = F.linear(qkv_input, self.k_weight, self.k_bias)
        value = F.linear(qkv_input, self.v_weight, self.v_bias)
        return query, key, value

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
            raise ValueError("Stage C attention wrapper expects post-LlamaModel position_embeddings for RoPE.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.spec.qjl_spec_key is None:
            kv_spec = STAGE_C_KV_SPECS[self.spec.kv_spec_key]
            attention = quantized_kv_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                kv_spec,
                signs=self.head_signs,
            )
        else:
            qjl_spec = STAGE_C_QJL_SPECS[self.spec.qjl_spec_key]
            attention = qjl_residual_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                self.num_key_value_groups,
                qjl_spec,
                seed=self.qjl_seed + int(self.layer_idx or 0),
                signs=self.head_signs,
            )

        attn_output = attention.output_heads.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        o_input, _ = quantize_stage_b_activation_domain(
            attn_output.to(dtype=self._linear_dtype()),
            self.spec.linear_spec,
            block_size=self.block_size,
        )
        output = F.linear(o_input, self.o_weight, self.o_bias).to(dtype=hidden_states.dtype)

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
                "key_hat": attention.key_hat.detach().cpu(),
                "value_hat": attention.value_hat.detach().cpu(),
                "output": output.detach().cpu(),
            }
        return output, attention.probs


def apply_stage_c_attention_fake_quant_(
    model: torch.nn.Module,
    method_name: str,
    *,
    block_size: int = 128,
    use_random_signs: bool = False,
    sign_seed: int = 0,
    qjl_seed: int = 0,
) -> list[dict[str, object]]:
    """Replace only self-attention modules with Stage C fake-quant wrappers."""
    if method_name not in STAGE_C_ATTENTION_SPECS:
        raise ValueError(f"Unknown Stage C model method: {method_name}")
    spec = STAGE_C_ATTENTION_SPECS[method_name]
    if spec.linear_spec is None:
        return []

    records: list[dict[str, object]] = []
    for layer_index, layer in iter_llama_decoder_layers(model):
        original_attention = layer.self_attn
        layer.self_attn = StageCAttentionWrapper(
            original_attention,
            method_name=method_name,
            block_size=block_size,
            use_random_signs=use_random_signs,
            sign_seed=sign_seed + layer_index,
            qjl_seed=qjl_seed,
        )
        method = STAGE_B_METHODS[spec.linear_spec.method]
        kv_spec = STAGE_C_KV_SPECS[spec.kv_spec_key]
        records.append(
            {
                "layer": f"model.layers.{layer_index}.self_attn",
                "method_key": method_name,
                "linear_method": method.name,
                "linear_bits": spec.linear_spec.label,
                "w_bits": spec.linear_spec.w_bits,
                "a_bits": spec.linear_spec.a_bits,
                "kv_method": kv_spec.method,
                "kv_bits": kv_spec.label,
                "k_bits": kv_spec.k_bits,
                "v_bits": kv_spec.v_bits,
                "qjl_method": spec.qjl_spec_key or "",
                "compute_interpretation": spec.compute_interpretation,
            }
        )
    return records


def fake_quant_attention_from_record(
    record,
    *,
    method_name: str,
    block_size: int = 128,
    use_random_signs: bool = False,
    sign_seed: int = 0,
    qjl_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one captured attention input through a Stage C attention wrapper."""
    wrapper = StageCAttentionWrapper(
        record.module,
        method_name=method_name,
        block_size=block_size,
        use_random_signs=use_random_signs,
        sign_seed=sign_seed + record.layer_index,
        qjl_seed=qjl_seed,
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
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return W_o R where R is the independent per-head H64 rotation.

    The weight shape is [out_features, hidden_size]. The input dimension is
    interpreted as concatenated attention heads; no cross-head mixing is used.
    """
    if weight.shape[-1] % head_dim != 0:
        raise ValueError(f"o_proj input dimension {weight.shape[-1]} is not divisible by head_dim={head_dim}.")
    shaped = weight.reshape(weight.shape[0], weight.shape[1] // head_dim, head_dim)
    absorbed = shaped
    if signs is not None:
        absorbed = absorbed * signs.to(device=weight.device, dtype=weight.dtype)
    return headwise_hadamard(absorbed, signs=None).reshape_as(weight)


class StageCRefineAttentionWrapper(torch.nn.Module):
    """Refined Stage C attention wrapper with optional value rotation absorption."""

    def __init__(
        self,
        attention_module: torch.nn.Module,
        *,
        method_name: str,
        block_size: int = 128,
        use_random_signs: bool = False,
        sign_seed: int = 0,
        record_details: bool = False,
    ) -> None:
        super().__init__()
        if method_name not in STAGE_C_REFINE_ATTENTION_SPECS:
            raise ValueError(f"Unknown Stage C refine attention method: {method_name}")
        self.method_name = method_name
        self.spec: StageCRefineAttentionSpec = STAGE_C_REFINE_ATTENTION_SPECS[method_name]
        self.block_size = block_size
        self.record_details = record_details
        self.config = attention_module.config
        self.layer_idx = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.num_key_value_groups = attention_module.num_key_value_groups
        self.scaling = float(attention_module.scaling)
        self.attention_dropout = getattr(attention_module, "attention_dropout", 0.0)
        self.last_details: dict[str, torch.Tensor] | None = None

        base_dtype = attention_module.q_proj.weight.dtype
        signs = (
            make_head_signs(
                self.head_dim,
                seed=sign_seed,
                device="cpu",
                dtype=base_dtype,
            )
            if use_random_signs
            else None
        )
        self.register_buffer("head_signs", signs)

        if self.spec.quantize_qkv:
            assert self.spec.linear_spec is not None
            q_weight, _ = prepare_stage_b_weight(attention_module.q_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
            k_weight, _ = prepare_stage_b_weight(attention_module.k_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
            v_weight, _ = prepare_stage_b_weight(attention_module.v_proj.weight.detach().cpu(), self.spec.linear_spec, block_size)
        else:
            q_weight = attention_module.q_proj.weight.detach().cpu()
            k_weight = attention_module.k_proj.weight.detach().cpu()
            v_weight = attention_module.v_proj.weight.detach().cpu()

        o_weight = attention_module.o_proj.weight.detach().cpu()
        if self.spec.value_path == "o_proj_absorb":
            o_weight = absorb_o_proj_weight_headwise(o_weight, head_dim=self.head_dim, signs=signs)
            if self.spec.quantize_o:
                assert self.spec.linear_spec is not None
                method = STAGE_B_METHODS[self.spec.linear_spec.method]
                o_weight, _ = quantize_stage_b_domain(
                    o_weight,
                    self.spec.linear_spec.w_bits,
                    method.quantizer,
                    block_size=self.head_dim,
                )
        elif self.spec.value_path == "reconstruct":
            if self.spec.quantize_o:
                assert self.spec.linear_spec is not None
                o_weight, _ = prepare_stage_b_weight(o_weight, self.spec.linear_spec, block_size)
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
                    block_size=self.head_dim,
                )
            else:
                o_input, _ = quantize_stage_b_activation_domain(
                    attn_output,
                    self.spec.linear_spec,
                    block_size=self.block_size,
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
            raise ValueError("Stage C refine wrapper expects post-LlamaModel position_embeddings for RoPE.")
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
            )
            pre_o_reference = inverse_headwise_hadamard(attention.output_heads, signs=self.head_signs)
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


def apply_stage_c_refine_attention_(
    model: torch.nn.Module,
    method_name: str,
    *,
    block_size: int = 128,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> list[dict[str, object]]:
    """Replace self-attention modules with refined Stage C wrappers."""
    if method_name not in STAGE_C_REFINE_ATTENTION_SPECS:
        raise ValueError(f"Unknown Stage C refine model method: {method_name}")
    spec = STAGE_C_REFINE_ATTENTION_SPECS[method_name]
    records: list[dict[str, object]] = []
    for layer_index, layer in iter_llama_decoder_layers(model):
        original_attention = layer.self_attn
        layer.self_attn = StageCRefineAttentionWrapper(
            original_attention,
            method_name=method_name,
            block_size=block_size,
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
                "value_path": spec.value_path,
                "compute_interpretation": spec.compute_interpretation,
            }
        )
    return records


def fake_quant_refine_attention_from_record(
    record,
    *,
    method_name: str,
    block_size: int = 128,
    use_random_signs: bool = False,
    sign_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one captured attention input through a refined Stage C wrapper."""
    wrapper = StageCRefineAttentionWrapper(
        record.module,
        method_name=method_name,
        block_size=block_size,
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
