# Stage C C5 PPL Summary

| method_key | linear_bits | kv_bits | block_size | kv_block_size | value_path | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 |  |  | reference | 8.048572581977634 | baseline |
| attn_rot_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | 32 | 64 | o_proj_absorb | 8.849057574428365 | Hadamard-rotated MXFP4 W4A4 linear fake quant plus HLM K4V4 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | 32 | 64 | o_proj_absorb | 8.621449658708212 | rotated LM W4A4 plus HLM K4V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | 32 | 64 | o_proj_absorb | 10.844038627893216 | rotated LM W3A4 plus HLM K3V4 with value rotation absorbed into o_proj |
