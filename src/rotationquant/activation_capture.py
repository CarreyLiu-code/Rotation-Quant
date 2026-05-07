from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rotationquant.modeling import iter_llama_decoder_layers, iter_llama_target_linears


@dataclass
class ActivationRecord:
    layer_index: int
    site: str
    tensor: torch.Tensor


@dataclass
class LinearIORecord:
    layer: str
    module: torch.nn.Linear
    input: torch.Tensor
    output: torch.Tensor


@dataclass
class FFNIORecord:
    layer: str
    module: torch.nn.Module
    input: torch.Tensor
    output: torch.Tensor


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _layer_index_from_name(name: str) -> int | None:
    parts = name.split(".")
    if "layers" not in parts:
        return None
    index = parts.index("layers") + 1
    if index >= len(parts):
        return None
    try:
        return int(parts[index])
    except ValueError:
        return None


class TinyLlamaActivationCapture:
    """Capture Stage B activation sites during one or more forward passes."""

    def __init__(self, model: torch.nn.Module, layer_limit: int | None = None) -> None:
        self.model = model
        self.layer_limit = layer_limit
        self.records: list[ActivationRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._ffn_temp: dict[int, dict[str, torch.Tensor]] = {}

    def __enter__(self) -> TinyLlamaActivationCapture:
        for layer_index, layer in iter_llama_decoder_layers(self.model):
            if self.layer_limit is not None and layer_index >= self.layer_limit:
                continue
            self._handles.append(layer.input_layernorm.register_forward_hook(self._capture(layer_index, "attn_input")))
            self._handles.append(
                layer.post_attention_layernorm.register_forward_hook(self._capture(layer_index, "ffn_input"))
            )
            self._handles.append(layer.self_attn.q_proj.register_forward_hook(self._capture(layer_index, "q_proj_out")))
            self._handles.append(layer.self_attn.k_proj.register_forward_hook(self._capture(layer_index, "k_proj_out")))
            self._handles.append(layer.self_attn.v_proj.register_forward_hook(self._capture(layer_index, "v_proj_out")))
            self._handles.append(layer.mlp.gate_proj.register_forward_hook(self._capture_ffn_gate(layer_index)))
            self._handles.append(layer.mlp.up_proj.register_forward_hook(self._capture_ffn_up(layer_index)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._ffn_temp.clear()

    def _capture(self, layer_index: int, site: str):
        def hook(_module, _inputs, output) -> None:
            tensor = _as_tensor(output).detach().cpu()
            self.records.append(ActivationRecord(layer_index=layer_index, site=site, tensor=tensor))

        return hook

    def _capture_ffn_gate(self, layer_index: int):
        def hook(_module, _inputs, output) -> None:
            self._ffn_temp.setdefault(layer_index, {})["gate"] = _as_tensor(output).detach().cpu()
            self._maybe_capture_ffn_intermediate(layer_index)

        return hook

    def _capture_ffn_up(self, layer_index: int):
        def hook(_module, _inputs, output) -> None:
            self._ffn_temp.setdefault(layer_index, {})["up"] = _as_tensor(output).detach().cpu()
            self._maybe_capture_ffn_intermediate(layer_index)

        return hook

    def _maybe_capture_ffn_intermediate(self, layer_index: int) -> None:
        temp = self._ffn_temp.get(layer_index, {})
        if "gate" not in temp or "up" not in temp:
            return
        intermediate = F.silu(temp.pop("gate")) * temp.pop("up")
        self.records.append(
            ActivationRecord(layer_index=layer_index, site="ffn_intermediate", tensor=intermediate.detach().cpu())
        )


class TinyLlamaLocalIOCapture:
    """Capture Linear and FFN input/output tensors for Stage B local experiments."""

    def __init__(self, model: torch.nn.Module, layer_limit: int | None = None) -> None:
        self.model = model
        self.layer_limit = layer_limit
        self.linear_records: list[LinearIORecord] = []
        self.ffn_records: list[FFNIORecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> TinyLlamaLocalIOCapture:
        for name, module in iter_llama_target_linears(self.model):
            layer_index = _layer_index_from_name(name)
            if self.layer_limit is not None and layer_index is not None and layer_index >= self.layer_limit:
                continue
            self._handles.append(module.register_forward_hook(self._capture_linear(name, module)))

        for layer_index, layer in iter_llama_decoder_layers(self.model):
            if self.layer_limit is not None and layer_index >= self.layer_limit:
                continue
            name = f"model.layers.{layer_index}.mlp"
            self._handles.append(layer.mlp.register_forward_hook(self._capture_ffn(name, layer.mlp)))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _capture_linear(self, name: str, module: torch.nn.Linear):
        def hook(_module, inputs, output) -> None:
            self.linear_records.append(
                LinearIORecord(
                    layer=name,
                    module=module,
                    input=_as_tensor(inputs[0]).detach().cpu(),
                    output=_as_tensor(output).detach().cpu(),
                )
            )

        return hook

    def _capture_ffn(self, name: str, module: torch.nn.Module):
        def hook(_module, inputs, output) -> None:
            self.ffn_records.append(
                FFNIORecord(
                    layer=name,
                    module=module,
                    input=_as_tensor(inputs[0]).detach().cpu(),
                    output=_as_tensor(output).detach().cpu(),
                )
            )

        return hook
