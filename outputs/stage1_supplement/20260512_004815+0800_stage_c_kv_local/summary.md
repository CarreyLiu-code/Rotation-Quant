# Stage C C2 KV Local Summary

| method_key | method | bits | kv_block_size | score_relative_mse | softmax_kl | topk_overlap | output_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | fp16 | K16V16 | nan | 0.0 | 0.0 | 1.0 | 1.00003 |
| hadamard_lm_k3v4 | hadamard_lm | K3V4 | 64.0 | 0.013198 | 0.035287 | 0.856783 | 0.958924 |
| hadamard_lm_k3v4_h32 | hadamard_lm | K3V4 | 32.0 | 0.012964 | 0.033944 | 0.860549 | 0.959275 |
| hadamard_lm_k4v4 | hadamard_lm | K4V4 | 64.0 | 0.003546 | 0.00951 | 0.918928 | 0.986786 |
| hadamard_lm_k4v4_h32 | hadamard_lm | K4V4 | 32.0 | 0.003405 | 0.009047 | 0.921434 | 0.9868 |
