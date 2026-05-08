# Stage C C2 KV Local Summary

| method_key | method | bits | score_relative_mse | softmax_kl | topk_overlap | output_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| absmax_k3v4 | absmax | K3V4 | 0.057856 | 0.810398 | 0.504865 | 0.908665 |
| absmax_k4v3 | absmax | K4V3 | 0.012673 | 0.238097 | 0.688388 | 0.957904 |
| absmax_k4v4 | absmax | K4V4 | 0.012673 | 0.238097 | 0.688388 | 0.97079 |
| fp16 | fp16 | K16V16 | 0.0 | 0.0 | 1.0 | 1.000012 |
| hadamard_lm_k2v4 | hadamard_lm | K2V4 | 0.042143 | 0.565414 | 0.534261 | 0.92724 |
| hadamard_lm_k3v3 | hadamard_lm | K3V3 | 0.011561 | 0.183433 | 0.690212 | 0.972211 |
| hadamard_lm_k3v4 | hadamard_lm | K3V4 | 0.011561 | 0.183433 | 0.690212 | 0.976442 |
| hadamard_lm_k4v3 | hadamard_lm | K4V3 | 0.00311 | 0.05205 | 0.818279 | 0.988012 |
| hadamard_lm_k4v4 | hadamard_lm | K4V4 | 0.00311 | 0.05205 | 0.818279 | 0.992406 |
