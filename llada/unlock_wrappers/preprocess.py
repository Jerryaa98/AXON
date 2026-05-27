"""Pure tensor transforms applied before the scheduler sees its inputs.

Each function is a no-op when the corresponding knob is inactive, so that
the wrapper's neutral configuration reproduces the original exactly.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def sparsify_attention(avg_attn_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Zero out attention entries below `threshold`. Returns a new tensor.

    Post-softmax attention is heavy-tailed; zeroing tiny entries before they
    reach the conflict matrix is a correctness-neutral speedup.
    threshold <= 0 is treated as a strict no-op (returns the input unchanged)."""
    if threshold is None or float(threshold) <= 0.0:
        return avg_attn_scores
    return torch.where(
        avg_attn_scores >= float(threshold),
        avg_attn_scores,
        torch.zeros_like(avg_attn_scores),
    )


@torch.no_grad()
def saliency_weighted_confidence(
    confidence: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
) -> torch.Tensor:
    """Rescale confidence at masked positions by a cheap saliency proxy.

    Saliency = normalized entropy of the softmax over the vocabulary. High
    entropy -> pivotal / undecided token -> we want the scheduler to treat
    its uncertainty as more important. Because the scheduler uses
    `target_uncertainty = 1 - conf_masked`, we shrink confidence toward zero
    at high-saliency positions so that (1 - c) grows there.

    Unmasked positions are returned unchanged. `logits` must be broadcastable
    to confidence (same leading dims; trailing dim is vocabulary)."""
    if confidence is None or logits is None:
        return confidence
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    eps = 1e-12
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    vocab = int(logits.shape[-1])
    norm = float(torch.log(torch.tensor(float(vocab))))
    saliency = (entropy / max(norm, eps)).clamp(0.0, 1.0).to(confidence.dtype)
    # Weight: saliency=0 -> keep confidence; saliency=1 -> drive confidence toward 0.
    weight = 1.0 - saliency
    weighted = confidence * weight
    return torch.where(mask_index, weighted, confidence)
