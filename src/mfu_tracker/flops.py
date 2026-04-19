"""FLOP counting formulas."""
from __future__ import annotations


def transformer_flops(
    num_params: int,
    tokens: int,
    *,
    with_backward: bool = True,
) -> int:
    """
    Estimate FLOPs for a transformer forward (or forward+backward) pass.

    Uses the standard 6ND approximation (Hoffmann et al., Chinchilla):
      forward  ≈ 2 * N * D
      backward ≈ 2x forward  →  total ≈ 6 * N * D

    Args:
        num_params: Total trainable parameter count.
        tokens:     Number of tokens processed in the step (batch * seq_len).
        with_backward: Include backward pass (True for training, False for inference).
    """
    multiplier = 6 if with_backward else 2
    return multiplier * num_params * tokens
