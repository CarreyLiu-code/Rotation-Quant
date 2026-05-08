# Stage C C5 PPL Summary

| method_key | linear_bits | kv_bits | qjl_method | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 |  | 617.3281523293532 | baseline |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 |  | 6.38453472796982 | rotated non-uniform W/A and KV fake quant |
