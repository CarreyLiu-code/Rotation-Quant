# Stage C C5 PPL Summary

| method_key | linear_bits | kv_bits | qjl_method | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 |  | 8.048572581977634 | baseline |
| attn_direct_absmax_w4a4_absmax_k4v4 | W4A4 | K4V4 |  | 2758.2516945450498 | direct W/A fake quant plus uniform KV fake quant |
| attn_rot_absmax_w4a4_hlm_k4v4 | W4A4 | K4V4 |  | 17607.464742118136 | rotated uniform W/A fake quant plus rotated KV fake quant |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 |  | 12044.78542229308 | rotated non-uniform W/A and KV fake quant |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 |  | 8105.302081889271 | weight and key bit reduction in fake quant |
| attn_rot_lm_w4a3_hlm_k4v3 | W4A3 | K4V3 |  | 8362.464386724363 | activation and value bit reduction in fake quant |
