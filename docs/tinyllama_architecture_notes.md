# TinyLlama Architecture Notes for Stage A

当前实验模型选择：

```text
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

本地路径：

```text
models/TinyLlama-1.1B-intermediate-step-1431k-3T
```

## Config 结论

已从 `config.json` 确认：

| Item | Value |
| --- | ---: |
| architecture | `LlamaForCausalLM` |
| layers | 22 |
| hidden size | 2048 |
| FFN intermediate size | 5632 |
| attention heads | 32 |
| KV heads | 4 |
| head dim | 64 |
| max position embeddings | 2048 |

TinyLlama 使用 GQA，所以 `k_proj` 和 `v_proj` 的输出维度是：

```text
num_key_value_heads * head_dim = 4 * 64 = 256
```

这意味着 A 阶段不同 Linear 权重 shape 不完全相同，尤其不能假设 q/k/v/o 都是 `[2048, 2048]`。

## A 阶段目标权重

按 Hugging Face LLaMA 命名，A 阶段只处理每个 decoder layer 内的这些 Linear 权重：

| Module suffix | Expected weight shape |
| --- | ---: |
| `self_attn.q_proj.weight` | `[2048, 2048]` |
| `self_attn.k_proj.weight` | `[256, 2048]` |
| `self_attn.v_proj.weight` | `[256, 2048]` |
| `self_attn.o_proj.weight` | `[2048, 2048]` |
| `mlp.gate_proj.weight` | `[5632, 2048]` |
| `mlp.up_proj.weight` | `[5632, 2048]` |
| `mlp.down_proj.weight` | `[2048, 5632]` |

共计：

```text
22 layers * 7 Linear weights = 154 target tensors
```

## A 阶段排除项

第一阶段保持 FP16，不参与 weight-only quantization：

```text
embed_tokens
input_layernorm
post_attention_layernorm
final norm
lm_head
```

## Block-wise Hadamard 适配

A 阶段默认 `block_size = 128`。上述所有目标 weight 的元素总数都能被 128 整除，因此理论上不需要 padding；代码仍保留 padding 逻辑，以便后续替换模型或扩展其他层时不脆弱。

## 下载完成后的复核

模型权重下载完成、`transformers` 安装后，运行：

```bash
PYTHONPATH=src conda run -n rotationquant python experiments/inspect_tinyllama_arch.py
```

该脚本会基于 `config.json` 输出目标 shape 和 block alignment；如果 `model.safetensors` 已下载，还会直接解析 safetensors header，复核真实权重 key 和 shape。

当前已复核结果：

```text
actual_target_weight_count = 154
```

真实 safetensors 权重命名与 A 阶段筛选规则一致，例如：

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
```
