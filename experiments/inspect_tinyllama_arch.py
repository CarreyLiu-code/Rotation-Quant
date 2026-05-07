from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

TINYLLAMA_BASE_DIR = "models/TinyLlama-1.1B-intermediate-step-1431k-3T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TinyLlama architecture for quantization planning.")
    parser.add_argument("--model-dir", default=TINYLLAMA_BASE_DIR)
    parser.add_argument("--output", default="outputs/model_arch/tinyllama_architecture.json")
    return parser.parse_args()


def expected_llama_linear_shapes(config: dict[str, object]) -> dict[str, list[int]]:
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim

    # Shapes follow torch.nn.Linear(out_features, in_features), hence the
    # stored weight tensor is [out_features, in_features].
    return {
        "self_attn.q_proj.weight": [hidden_size, hidden_size],
        "self_attn.k_proj.weight": [kv_dim, hidden_size],
        "self_attn.v_proj.weight": [kv_dim, hidden_size],
        "self_attn.o_proj.weight": [hidden_size, hidden_size],
        "mlp.gate_proj.weight": [intermediate_size, hidden_size],
        "mlp.up_proj.weight": [intermediate_size, hidden_size],
        "mlp.down_proj.weight": [hidden_size, intermediate_size],
    }


def read_safetensors_header(path: Path) -> dict[str, object]:
    """Read safetensors metadata without importing torch or safetensors."""
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = f.read(header_size)
    return json.loads(header)


def actual_target_shapes_from_safetensors(model_dir: Path) -> dict[str, list[int]]:
    weight_path = model_dir / "model.safetensors"
    if not weight_path.exists():
        return {}
    header = read_safetensors_header(weight_path)
    actual: dict[str, list[int]] = {}
    suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if key.endswith(suffixes):
            actual[key] = value["shape"]
    return dict(sorted(actual.items()))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    shapes = expected_llama_linear_shapes(config)
    actual_target_shapes = actual_target_shapes_from_safetensors(model_dir)
    block_size = 128
    report = {
        "model_dir": str(model_dir),
        "architecture": config.get("architectures", ["unknown"])[0],
        "model_type": config.get("model_type"),
        "num_hidden_layers": config["num_hidden_layers"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config.get("num_key_value_heads", config["num_attention_heads"]),
        "head_dim": int(config["hidden_size"]) // int(config["num_attention_heads"]),
        "target_weight_shapes": shapes,
        "actual_target_weight_count": len(actual_target_shapes),
        "actual_target_weight_shapes": actual_target_shapes,
        "stage_a_block_size": block_size,
        "block_alignment": {
            name: (shape[0] * shape[1]) % block_size == 0 for name, shape in shapes.items()
        },
        "stage_a_exclusions": ["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm", "lm_head"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
