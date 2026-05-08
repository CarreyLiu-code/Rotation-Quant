# Stage C Refine PPL Summary

| method_key | linear_bits | kv_bits | value_path | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 | reference | 8.048572581977634 | baseline |
| attn_identity_fp16 | FP16 | K16V16 | reconstruct | 8.048572581977634 | identity attention wrapper; no quantization |
| attn_kv_only_hlm_k4v4_oabsorb | FP16 | K4V4 | o_proj_absorb | 8.204481572257707 | KV-only HLM K4V4 with value rotation absorbed into o_proj |
| attn_kv_only_hlm_k3v4_oabsorb | FP16 | K3V4 | o_proj_absorb | 8.658396294517212 | KV-only HLM K3V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w4a4_hlm_k4v4_oabsorb | W4A4 | K4V4 | o_proj_absorb | 8.61449660535375 | rotated LM W4A4 plus HLM K4V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w3a4_hlm_k3v4_oabsorb | W3A4 | K3V4 | o_proj_absorb | 11.355422325915786 | rotated LM W3A4 plus HLM K3V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w4a3_hlm_k4v3_oabsorb | W4A3 | K4V3 | o_proj_absorb | 9.387664799222907 | rotated LM W4A3 plus HLM K4V3 with value rotation absorbed into o_proj |
