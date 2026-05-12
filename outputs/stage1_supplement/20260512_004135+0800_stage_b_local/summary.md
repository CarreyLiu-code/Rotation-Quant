# Stage B B2 Local Summary

## Linear

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| mxfp4_w4a4 | mxfp4 | W4A4 | 0.02444 | 0.98819 | 17.202456 |
| rot_lm_w3a4 | rot_lm | W3A4 | 0.037209 | 0.982336 | 16.415885 |
| rot_lm_w4a4 | rot_lm | W4A4 | 0.015854 | 0.992299 | 20.221924 |
| rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.022691 | 0.989112 | 18.590158 |

## FFN

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| ffn_fp16 | fp16 | FP16 | 0.0 | 1.000025 | 96.06341 |
| ffn_mxfp4_w4a4 | mxfp4 | W4A4 | 0.059908 | 0.97272 | 12.332545 |
| ffn_rot_lm_w3a4 | rot_lm | W3A4 | 0.073063 | 0.964408 | 11.562464 |
| ffn_rot_lm_w4a4 | rot_lm | W4A4 | 0.031212 | 0.984782 | 15.330151 |
| ffn_rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.045629 | 0.977517 | 13.724342 |
