from __future__ import annotations

import math

import torch
from tqdm import tqdm


def load_text_dataset(dataset_name: str, dataset_config: str, split: str, text_column: str = "text"):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for PPL evaluation.") from exc
    return load_dataset(dataset_name, dataset_config, split=split)[text_column]


def tokenize_texts(tokenizer, texts: list[str], max_samples: int | None = None) -> torch.Tensor:
    if max_samples is not None:
        texts = texts[:max_samples]
    joined = "\n\n".join(text for text in texts if text)
    encoded = tokenizer(joined, return_tensors="pt")
    return encoded.input_ids


@torch.no_grad()
def evaluate_causal_lm_ppl(
    model,
    input_ids: torch.Tensor,
    sequence_length: int = 2048,
    stride: int = 2048,
    device: str | torch.device | None = None,
) -> float:
    """Sliding-window causal LM perplexity.

    The implementation masks context tokens with -100, so each token is scored
    once even when `stride < sequence_length`.
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    negative_log_likelihoods: list[torch.Tensor] = []
    previous_end = 0
    for begin in tqdm(range(0, input_ids.size(1), stride), desc="PPL windows"):
        end = min(begin + sequence_length, input_ids.size(1))
        target_length = end - previous_end
        window = input_ids[:, begin:end]
        targets = window.clone()
        targets[:, :-target_length] = -100
        outputs = model(window, labels=targets)
        negative_log_likelihoods.append(outputs.loss.detach())
        previous_end = end
        if end == input_ids.size(1):
            break

    mean_nll = torch.stack(negative_log_likelihoods).mean().cpu()
    if not torch.isfinite(mean_nll):
        return float("nan")
    # Very broken quantization settings can produce a valid but enormous NLL.
    # Record the failure as infinite PPL instead of aborting the whole sweep.
    mean_nll_float = float(mean_nll)
    if mean_nll_float > 700.0:
        return float("inf")
    return float(math.exp(mean_nll_float))
