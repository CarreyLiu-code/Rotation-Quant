# Stage C C4 Attention Layer Summary

| method_key | linear_bits | kv_bits | projection_relative_mse | score_relative_mse | softmax_kl | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attn_direct_absmax_w4a4_absmax_k4v4 | W4A4 | K4V4 | 0.363401 | 0.128626 | 0.599042 | 1.314227 | 0.250457 |
| attn_rot_absmax_w4a4_hlm_k4v4 | W4A4 | K4V4 | 0.143706 | 0.043899 | 0.160127 | 1.618342 | 0.597002 |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | 0.054862 | 0.018479 | 0.187293 | 0.287546 | 0.864589 |
| attn_rot_lm_w4a3_hlm_k4v3 | W4A3 | K4V3 | 0.053374 | 0.00858 | 0.072355 | 0.186937 | 0.907371 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | 0.023228 | 0.005036 | 0.057384 | 0.105883 | 0.947851 |
| fp16 | FP16 | K16V16 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
