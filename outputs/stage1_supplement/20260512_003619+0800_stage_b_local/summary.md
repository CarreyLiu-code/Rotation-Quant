# Stage B B2 Local Summary

## Linear

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| mxfp4_w4a4 | mxfp4 | W4A4 | 0.02444 | 0.98819 | 17.202456 |
| rot_lm_w3a4 | rot_lm | W3A4 | 0.034288 | 0.983574 | 16.66345 |
| rot_lm_w4a4 | rot_lm | W4A4 | 0.014332 | 0.992999 | 20.449404 |
| rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.021605 | 0.989527 | 18.535096 |

## FFN

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| ffn_fp16 | fp16 | FP16 | 0.0 | 1.000025 | 96.06341 |
| ffn_mxfp4_w4a4 | mxfp4 | W4A4 | 0.059908 | 0.97272 | 12.332545 |
| ffn_rot_lm_w3a4 | rot_lm | W3A4 | 0.069378 | 0.965928 | 11.787744 |
| ffn_rot_lm_w4a4 | rot_lm | W4A4 | 0.029818 | 0.985446 | 15.517956 |
| ffn_rot_mxfp4_w4a4 | rot_mxfp4 | W4A4 | 0.045747 | 0.977814 | 13.620184 |
