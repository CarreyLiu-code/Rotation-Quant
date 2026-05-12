# Stage C C4 Attention-layer Summary

| method_key | linear_bits | kv_bits | value_path | block_size | kv_block_size | score_relative_mse | softmax_kl | pre_o_output_cosine | layer_output_relative_mse | layer_output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attn_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 32 | 64.0 | 0.012066 | 0.024538 | 0.941626 | 0.12004 | 0.942516 |
| attn_rot_lm_w3a4_hlm_k3v4 | W3A4 | K3V4 | o_proj_absorb | 32 | 64.0 | 0.019489 | 0.046539 | 0.856349 | 0.246934 | 0.88602 |
| attn_rot_lm_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 32 | 64.0 | 0.005681 | 0.013871 | 0.941805 | 0.086423 | 0.958561 |
| attn_rot_mxfp4_w4a4_hlm_k4v4 | W4A4 | K4V4 | o_proj_absorb | 32 | 64.0 | 0.007534 | 0.016426 | 0.923789 | 0.118463 | 0.94283 |
| fp16 | FP16 | K16V16 | reference |  | nan | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
