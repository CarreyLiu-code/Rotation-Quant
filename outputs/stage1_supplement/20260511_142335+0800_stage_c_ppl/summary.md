# Stage C C5 PPL Summary

| method_key | linear_bits | kv_bits | value_path | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 | reference | 8.048572581977634 | baseline |
| attn_kv_hlm_k4v4 | FP16 | K4V4 | o_proj_absorb | 8.204481572257707 | KV-only HLM K4V4 with value rotation absorbed into o_proj |
| attn_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 8.995300208592932 | MXFP4 W4A4 linear fake quant plus HLM K4V4 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 8.61449660535375 | rotated LM W4A4 plus HLM K4V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 11.355422325915786 | rotated LM W3A4 plus HLM K3V4 with value rotation absorbed into o_proj |
| attn_randhadamard_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 8.636056165782495 | randomized Hadamard LM W4A4 plus randomized HLM K4V4 |
| attn_randortho_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 8.607538898527151 | dense random orthogonal LM W4A4 plus random-orthogonal K4V4 |
