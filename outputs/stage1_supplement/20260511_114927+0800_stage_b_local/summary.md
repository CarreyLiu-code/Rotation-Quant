# Stage B B2 Local Summary

## Linear

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| direct_absmax_w4a4 | direct_absmax | W4A4 | 0.044481 | 0.978356 | 15.493451 |
| mxfp4_w4a4 | mxfp4 | W4A4 | 0.02444 | 0.98819 | 17.202456 |
| randhadamard_lm_w3a4 | randhadamard_lm | W3A4 | 0.036574 | 0.982611 | 16.45432 |
| randhadamard_lm_w4a3 | randhadamard_lm | W4A3 | 0.036528 | 0.983207 | 16.449216 |
| randhadamard_lm_w4a4 | randhadamard_lm | W4A4 | 0.01547 | 0.992478 | 20.329174 |
| randortho_lm_w3a4 | randortho_lm | W3A4 | 0.037956 | 0.982161 | 16.335459 |
| randortho_lm_w4a3 | randortho_lm | W4A3 | 0.037246 | 0.982412 | 16.366434 |
| randortho_lm_w4a4 | randortho_lm | W4A4 | 0.016069 | 0.99221 | 20.188213 |
| rot_absmax_w4a4 | rot_absmax | W4A4 | 0.021578 | 0.989686 | 18.903518 |
| rot_lm_w3a3 | rot_lm | W3A3 | 0.058215 | 0.973365 | 14.323914 |
| rot_lm_w3a4 | rot_lm | W3A4 | 0.037209 | 0.982336 | 16.415885 |
| rot_lm_w4a3 | rot_lm | W4A3 | 0.037381 | 0.982784 | 16.350577 |
| rot_lm_w4a4 | rot_lm | W4A4 | 0.015854 | 0.992299 | 20.221924 |

## FFN

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| ffn_direct_absmax_w4a4 | direct_absmax | W4A4 | 0.086249 | 0.959065 | 11.802458 |
| ffn_fp16 | fp16 | FP16 | 0.0 | 1.000025 | 96.06341 |
| ffn_mxfp4_w4a4 | mxfp4 | W4A4 | 0.059908 | 0.97272 | 12.332545 |
| ffn_randhadamard_lm_w3a4 | randhadamard_lm | W3A4 | 0.073021 | 0.964347 | 11.578339 |
| ffn_randhadamard_lm_w4a3 | randhadamard_lm | W4A3 | 0.07772 | 0.964415 | 11.188968 |
| ffn_randhadamard_lm_w4a4 | randhadamard_lm | W4A4 | 0.030943 | 0.984823 | 15.430528 |
| ffn_randortho_lm_w3a4 | randortho_lm | W3A4 | 0.073837 | 0.963857 | 11.569227 |
| ffn_randortho_lm_w4a3 | randortho_lm | W4A3 | 0.074992 | 0.963041 | 11.51853 |
| ffn_randortho_lm_w4a4 | randortho_lm | W4A4 | 0.031507 | 0.984347 | 15.365829 |
| ffn_rot_absmax_w4a4 | rot_absmax | W4A4 | 0.044207 | 0.978834 | 14.140494 |
| ffn_rot_lm_w3a3 | rot_lm | W3A3 | 0.121197 | 0.944023 | 9.253756 |
| ffn_rot_lm_w3a4 | rot_lm | W3A4 | 0.073063 | 0.964408 | 11.562464 |
| ffn_rot_lm_w4a3 | rot_lm | W4A3 | 0.079189 | 0.964333 | 11.106795 |
| ffn_rot_lm_w4a4 | rot_lm | W4A4 | 0.031212 | 0.984782 | 15.330151 |
