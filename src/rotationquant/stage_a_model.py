from __future__ import annotations

import torch

from rotationquant.modeling import iter_llama_target_linears
from rotationquant.stage_a import quantize_weight_for_stage_a


def apply_stage_a_weight_quant_(
    model: torch.nn.Module,
    bits: int,
    method_name: str,
    block_size: int = 128,
    mxfp4_group_size: int = 32,
    rotation_seed: int = 0,
) -> list[dict[str, object]]:
    """Quantize Stage A target Linear weights in-place and return metadata.

    This keeps activations, embeddings, norms, and lm_head unchanged, matching
    the A16Wb model-level setting in the experiment plan.
    """
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for layer_name, layer in iter_llama_target_linears(model):
            # Quantize on CPU for broad operator coverage, then copy the
            # dequantized fake-quant weight back to the model device.
            weight_cpu = layer.weight.detach().cpu()
            quantized_weight, metadata = quantize_weight_for_stage_a(
                weight_cpu,
                bits=bits,
                method_name=method_name,
                block_size=block_size,
                mxfp4_group_size=mxfp4_group_size,
                rotation_seed=rotation_seed,
            )
            layer.weight.copy_(quantized_weight.to(device=layer.weight.device, dtype=layer.weight.dtype))
            records.append({"layer": layer_name, **metadata})
    return records
