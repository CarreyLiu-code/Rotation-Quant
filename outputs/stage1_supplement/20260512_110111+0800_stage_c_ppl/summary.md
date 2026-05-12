# Stage C C5 PPL Summary

| method_key | linear_bits | kv_bits | block_size | kv_block_size | value_path | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attn_rot_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | 128 | 64 | o_proj_absorb | 8.933362568844736 | Hadamard-rotated MXFP4 W4A4 linear fake quant plus HLM K4V4 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | 128 | 64 | o_proj_absorb | 8.61449660535375 | rotated LM W4A4 plus HLM K4V4 with value rotation absorbed into o_proj |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | 128 | 64 | o_proj_absorb | 11.355422325915786 | rotated LM W3A4 plus HLM K3V4 with value rotation absorbed into o_proj |
