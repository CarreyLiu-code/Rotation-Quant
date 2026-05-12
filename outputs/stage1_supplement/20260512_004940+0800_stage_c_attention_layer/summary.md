# Stage C C4 Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | block_size | kv_block_size | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 64 | 64.0 | 0.012066 | 0.024538 | 0.941626 | 0.12004 | 0.942516 |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 64 | 64.0 | 0.020134 | 0.045804 | 0.855587 | 0.252072 | 0.884712 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 64 | 64.0 | 0.005701 | 0.013419 | 0.940866 | 0.084833 | 0.958515 |
| attn_rot_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 64 | 64.0 | 0.00745 | 0.016204 | 0.922447 | 0.118201 | 0.942786 |
| fp16 | FP16 | K16V16 | reference |  | nan | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
