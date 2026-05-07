from __future__ import annotations

from collections.abc import Iterator

import torch


TINYLLAMA_BASE_DIR = "models/TinyLlama-1.1B-intermediate-step-1431k-3T"

LLAMA_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def iter_llama_target_linears(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Linear]]:
    """Yield only the q/k/v/o and FFN linears targeted by Stage A."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(LLAMA_LINEAR_SUFFIXES):
            yield name, module


def iter_llama_decoder_layers(model: torch.nn.Module) -> Iterator[tuple[int, torch.nn.Module]]:
    """Yield decoder layers from Hugging Face LLaMA-like causal LM models."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Expected a Hugging Face LLaMA-like model with model.layers.")
    for index, layer in enumerate(layers):
        yield index, layer


def iter_llama_ffn_modules(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield FFN modules; Hugging Face calls them `mlp`, but Stage B uses FFN."""
    for index, layer in iter_llama_decoder_layers(model):
        ffn = getattr(layer, "mlp", None)
        if ffn is None:
            raise ValueError(f"Decoder layer {index} has no mlp/FFN module.")
        yield f"model.layers.{index}.mlp", ffn


def load_causal_lm(model_dir: str, dtype: str = "float16", device_map: str | None = None):
    """Load a local Hugging Face causal LM without tying the code to one checkpoint."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load TinyLlama. Install project requirements first."
        ) from exc

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    load_kwargs = {
        "device_map": device_map,
        # Keep experiments reproducible/offline once the snapshot is downloaded.
        "local_files_only": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, **load_kwargs)
    except TypeError:
        # Older Transformers releases used torch_dtype. Keeping the fallback
        # makes the scripts portable across lab machines.
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, **load_kwargs)
    model.eval()
    return model, tokenizer
