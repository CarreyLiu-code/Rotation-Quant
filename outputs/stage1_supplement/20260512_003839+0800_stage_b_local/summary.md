# Stage B B2 Local Summary

## Linear

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| mxfp4_w4a4 | mxfp4 | W4A4 | 0.02444 | 0.98819 | 17.202456 |
| rot_lm_w3a4 | rot_lm | W3A4 | 0.035734 | 0.983 | 16.515835 |
| rot_lm_w4a4 | rot_lm | W4A4 | 0.014696 | 0.992814 | 20.355546 |
| rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.022084 | 0.989413 | 18.490542 |

## FFN

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| ffn_fp16 | fp16 | FP16 | 0.0 | 1.000025 | 96.06341 |
| ffn_mxfp4_w4a4 | mxfp4 | W4A4 | 0.059908 | 0.97272 | 12.332545 |
| ffn_rot_lm_w3a4 | rot_lm | W3A4 | 0.072402 | 0.964819 | 11.570336 |
| ffn_rot_lm_w4a4 | rot_lm | W4A4 | 0.030791 | 0.985009 | 15.374483 |
| ffn_rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.047259 | 0.977614 | 13.383885 |
