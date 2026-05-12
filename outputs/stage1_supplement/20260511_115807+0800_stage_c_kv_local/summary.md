# Stage C C2 KV Local Summary

| method_key | method | bits | score_relative_mse | softmax_kl | topk_overlap | output_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| absmax_k4v4 | absmax | K4V4 | 0.015507 | 0.046387 | 0.847715 | 0.944755 |
| fp16 | fp16 | K16V16 | 0.0 | 0.0 | 1.0 | 1.00003 |
| hadamard_lm_k3v4 | hadamard_lm | K3V4 | 0.013198 | 0.035287 | 0.856783 | 0.958924 |
| hadamard_lm_k4v3 | hadamard_lm | K4V3 | 0.003546 | 0.00951 | 0.918928 | 0.978658 |
| hadamard_lm_k4v4 | hadamard_lm | K4V4 | 0.003546 | 0.00951 | 0.918928 | 0.986786 |
| randhadamard_lm_k3v4 | hadamard_lm | K3V4 | 0.013522 | 0.035206 | 0.85715 | 0.958748 |
| randhadamard_lm_k4v3 | hadamard_lm | K4V3 | 0.003731 | 0.00937 | 0.918667 | 0.979111 |
| randhadamard_lm_k4v4 | hadamard_lm | K4V4 | 0.003731 | 0.00937 | 0.918667 | 0.987079 |
| randortho_lm_k3v4 | hadamard_lm | K3V4 | 0.014519 | 0.038788 | 0.853449 | 0.955308 |
| randortho_lm_k4v3 | hadamard_lm | K4V3 | 0.004014 | 0.010676 | 0.915991 | 0.976827 |
| randortho_lm_k4v4 | hadamard_lm | K4V4 | 0.004014 | 0.010676 | 0.915991 | 0.985192 |
