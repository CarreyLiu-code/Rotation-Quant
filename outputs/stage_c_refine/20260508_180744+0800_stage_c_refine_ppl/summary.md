# Stage C Refine PPL Summary

| method_key | linear_bits | kv_bits | value_path | ppl | compute_interpretation |
| --- | --- | --- | --- | --- | --- |
| fp16 | FP16 | K16V16 | reference | 617.3281523293532 | baseline |
| attn_identity_fp16 | FP16 | K16V16 | reconstruct | 4.842428256349744 | identity attention wrapper; no quantization |
| attn_kv_only_hlm_k4v4_oabsorb | FP16 | K4V4 | o_proj_absorb | 2.7808253025078735 | KV-only HLM K4V4 with value rotation absorbed into o_proj |
