from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotationquant.modeling import iter_llama_decoder_layers
from rotationquant.stage_c import attention_probs, attention_output_from_probs, raw_attention_inner_product

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - import is validated by smoke scripts.
    raise RuntimeError("Stage C attention capture requires Hugging Face LLaMA modules.") from exc


@dataclass
class AttentionIORecord:
    layer_index: int
    layer: str
    module: torch.nn.Module
    input: torch.Tensor
    q_proj_out: torch.Tensor
    k_proj_out: torch.Tensor
    v_proj_out: torch.Tensor
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    attention_mask: torch.Tensor | None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    attn_probs: torch.Tensor
    attn_output_heads: torch.Tensor
    output: torch.Tensor
    scaling: float
    num_key_value_groups: int


class _CaptureAttentionWrapper(torch.nn.Module):
    """Temporarily replace LlamaAttention so RoPE-local tensors can be recorded."""

    def __init__(
        self,
        original_attention: torch.nn.Module,
        *,
        layer_index: int,
        records: list[AttentionIORecord],
    ) -> None:
        super().__init__()
        self.original_attention = original_attention
        self.layer_index = layer_index
        self.records = records
        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        self.head_dim = original_attention.head_dim
        self.num_key_value_groups = original_attention.num_key_value_groups
        self.scaling = float(original_attention.scaling)
        self.attention_dropout = getattr(original_attention, "attention_dropout", 0.0)

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

        query_states = self.original_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.original_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q_proj_out = query_states
        k_proj_out = key_states

        if position_embeddings is None:
            raise ValueError("Stage C capture expects post-LlamaModel position_embeddings for RoPE.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        raw_ip = raw_attention_inner_product(query_states, key_states, self.num_key_value_groups)
        scores = raw_ip * self.scaling
        attn_weights = attention_probs(scores, attention_mask)
        if self.training and self.attention_dropout:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
        attn_output_heads = attention_output_from_probs(attn_weights, value_states, self.num_key_value_groups)
        attn_output = attn_output_heads.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attention.o_proj(attn_output)

        self.records.append(
            AttentionIORecord(
                layer_index=self.layer_index,
                layer=f"model.layers.{self.layer_index}.self_attn",
                module=self.original_attention,
                input=hidden_states.detach().cpu(),
                q_proj_out=q_proj_out.detach().cpu(),
                k_proj_out=k_proj_out.detach().cpu(),
                v_proj_out=value_states.detach().cpu(),
                q_rope=query_states.detach().cpu(),
                k_rope=key_states.detach().cpu(),
                attention_mask=attention_mask.detach().cpu() if attention_mask is not None else None,
                position_embeddings=(cos.detach().cpu(), sin.detach().cpu()),
                attn_probs=attn_weights.detach().cpu(),
                attn_output_heads=attn_output_heads.detach().cpu(),
                output=attn_output.detach().cpu(),
                scaling=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
            )
        )
        return attn_output, attn_weights


class TinyLlamaAttentionCapture:
    """Capture Stage C attention internals by temporarily wrapping self-attention."""

    def __init__(self, model: torch.nn.Module, layer_limit: int | None = None) -> None:
        self.model = model
        self.layer_limit = layer_limit
        self.records: list[AttentionIORecord] = []
        self._originals: list[tuple[torch.nn.Module, torch.nn.Module]] = []

    def __enter__(self) -> TinyLlamaAttentionCapture:
        for layer_index, layer in iter_llama_decoder_layers(self.model):
            if self.layer_limit is not None and layer_index >= self.layer_limit:
                continue
            original = layer.self_attn
            layer.self_attn = _CaptureAttentionWrapper(
                original,
                layer_index=layer_index,
                records=self.records,
            )
            self._originals.append((layer, original))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for layer, original in self._originals:
            layer.self_attn = original
        self._originals.clear()
