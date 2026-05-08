# Stage C Refine Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_identity_fp16 | FP16 | K16V16 | reconstruct | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| attn_kv_only_hlm_k4v4_oabsorb | FP16 | K4V4 | o_proj_absorb | 0.007055 | 0.002684 | 0.992694 | 0.005099 | 0.99757 |
| attn_kv_only_hlm_k4v4_reconstruct | FP16 | K4V4 | reconstruct | 0.007055 | 0.002684 | 0.992694 | 0.005098 | 0.997571 |
| attn_rot_lm_w4a4_hlm_k4v4_oabsorb | W4A4 | K4V4 | o_proj_absorb | 0.010309 | 0.004012 | 0.879333 | 0.044996 | 0.97806 |
| fp16 | FP16 | K16V16 | reference | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
