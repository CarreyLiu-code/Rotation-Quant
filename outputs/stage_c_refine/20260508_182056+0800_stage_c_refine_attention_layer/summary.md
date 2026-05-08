# Stage C Refine Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_identity_fp16 | FP16 | K16V16 | reconstruct | 0.0 | 0.0 | 1.000026 | 1e-06 | 1.000005 |
| attn_kv_only_hlm_k3v4_oabsorb | FP16 | K3V4 | o_proj_absorb | 0.013185 | 0.035261 | 0.95901 | 0.083876 | 0.961825 |
| attn_kv_only_hlm_k4v3_oabsorb | FP16 | K4V3 | o_proj_absorb | 0.003558 | 0.00954 | 0.978642 | 0.036435 | 0.981916 |
| attn_kv_only_hlm_k4v4_oabsorb | FP16 | K4V4 | o_proj_absorb | 0.003558 | 0.00954 | 0.986765 | 0.024623 | 0.988034 |
| attn_kv_only_hlm_k4v4_reconstruct | FP16 | K4V4 | reconstruct | 0.003558 | 0.00954 | 0.986765 | 0.024622 | 0.988034 |
| attn_rot_lm_w3a4_hlm_k3v4_oabsorb | W3A4 | K3V4 | o_proj_absorb | 0.020985 | 0.046599 | 0.851703 | 0.2558 | 0.881672 |
| attn_rot_lm_w4a3_hlm_k4v3_oabsorb | W4A3 | K4V3 | o_proj_absorb | 0.010173 | 0.019506 | 0.903753 | 0.162295 | 0.920949 |
| attn_rot_lm_w4a4_hlm_k4v4_oabsorb | W4A4 | K4V4 | o_proj_absorb | 0.005677 | 0.013771 | 0.936548 | 0.089098 | 0.956899 |
| fp16 | FP16 | K16V16 | reference | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
