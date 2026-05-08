# Stage C Refine Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_identity_fp16 | FP16 | K16V16 | reconstruct | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| attn_kv_only_hlm_k4v4_oabsorb | FP16 | K4V4 | o_proj_absorb | 0.007055 | 0.002513 | 0.992267 | 0.00383 | 0.998133 |
| attn_kv_only_hlm_k4v4_reconstruct | FP16 | K4V4 | reconstruct | 0.007055 | 0.002513 | 0.992267 | 0.003829 | 0.998133 |
| attn_rot_lm_w4a4_hlm_k4v4_oabsorb | W4A4 | K4V4 | o_proj_absorb | 0.010309 | 0.005061 | 0.903032 | 0.03754 | 0.982361 |
| fp16 | FP16 | K16V16 | reference | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
