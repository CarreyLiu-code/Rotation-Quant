# Stage C C4 Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | block_size | kv_block_size | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 128 | 64.0 | 0.012066 | 0.024538 | 0.941626 | 0.12004 | 0.942516 |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 128 | 64.0 | 0.020985 | 0.046599 | 0.851703 | 0.2558 | 0.881672 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 128 | 64.0 | 0.005677 | 0.013771 | 0.936548 | 0.089098 | 0.956899 |
| attn_rot_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 128 | 64.0 | 0.007047 | 0.016034 | 0.920325 | 0.12235 | 0.941342 |
| fp16 | FP16 | K16V16 | reference |  | nan | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
