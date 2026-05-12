# Stage C C4 Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_kv_hlm_k4v4 | FP16 | K4V4 | o_proj_absorb | 0.003558 | 0.00954 | 0.986765 | 0.024623 | 0.988034 |
| attn_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 0.012066 | 0.024538 | 0.941626 | 0.12004 | 0.942516 |
| attn_randhadamard_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 0.021447 | 0.046927 | 0.852947 | 0.248529 | 0.884771 |
| attn_randhadamard_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 0.005769 | 0.013588 | 0.935738 | 0.08952 | 0.956743 |
| attn_randortho_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 0.026691 | 0.057621 | 0.840661 | 0.284049 | 0.873701 |
| attn_randortho_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 0.006916 | 0.01572 | 0.93147 | 0.098507 | 0.953112 |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 0.020985 | 0.046599 | 0.851703 | 0.2558 | 0.881672 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 0.005677 | 0.013771 | 0.936548 | 0.089098 | 0.956899 |
| fp16 | FP16 | K16V16 | reference | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
