"""FLOP counting — FlopCounterMode (PyTorch 2.1+) with thop fallback."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def flash_attn_flops(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    *,
    causal: bool = True,
    with_backward: bool = False,
) -> int:
    """
    Analytical FLOP count for one flash attention call (escape hatch).

    ``profile_flops()`` already counts ``F.scaled_dot_product_attention``
    automatically on CUDA via ``FlopCounterMode``. Use this only if your model
    calls the ``flash_attn`` C extension directly (``flash_attn_func``), which
    is rare in modern codebases.

    Formula:
        causal=True  → 2 * batch * seq_len² * num_heads * head_dim FLOPs
        causal=False → 4 * batch * seq_len² * num_heads * head_dim FLOPs
    """
    fwd_flops = 2 * batch * seq_len * seq_len * num_heads * head_dim
    if not causal:
        fwd_flops *= 2
    return fwd_flops * 3 if with_backward else fwd_flops


def _profile_with_flop_counter(target: nn.Module, call_args: tuple) -> int:
    """Profile using torch.utils.flop_counter.FlopCounterMode (PyTorch 2.1+).

    Works at the ATen dispatch level — counts F.scaled_dot_product_attention
    (SDPA / native flash attention) on CUDA automatically. Returns -1 on failure.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return -1

    was_training = target.training
    target.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
            target(*call_args)
        return int(flop_counter.get_total_flops())
    except Exception:
        return -1
    finally:
        target.train(was_training)


def _profile_with_thop(target: nn.Module, call_args: tuple) -> int:
    """Profile using thop — fallback for PyTorch < 2.1 or unsupported models."""
    from thop import profile

    macs, _ = profile(target, inputs=call_args, verbose=False)
    return int(macs * 2)


def profile_flops(
    model: nn.Module,
    args: Optional[tuple] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    with_backward: bool = True,
) -> int:
    """
    Count FLOPs for one forward (or forward+backward) pass through *model*.

    Uses ``torch.utils.flop_counter.FlopCounterMode`` (PyTorch 2.1+), which
    hooks at the ATen operator level and automatically counts
    ``F.scaled_dot_product_attention`` (SDPA) on CUDA — covering virtually all
    modern transformer attention implementations without any manual correction.
    Falls back to ``thop`` on older PyTorch versions.

    Note: SDPA FLOPs are only counted when the model is on CUDA (the kernel
    dispatches differently on CPU). If you profile a CPU model, attention FLOPs
    will be missing — move the model to CUDA first for an accurate count.

    Args:
        model:         The nn.Module to profile.
        args:          Positional inputs passed to model(*args).
        kwargs:        Keyword-only inputs (e.g. HF models). Baked into a thin
                       wrapper since both profilers call ``model(*inputs)``.
        with_backward: Include backward-pass FLOPs (default True for training).
                       Backward ≈ 2× forward → 3× total.

    Returns:
        Total integer FLOP count for one step.
    """
    # kwargs-only models need a wrapper — both profilers call target(*call_args)
    if kwargs:
        _kw = kwargs

        class _KwargsAdapter(nn.Module):
            def forward(self):
                return model(**(args[0] if args and isinstance(args[0], dict) else _kw))

        target: nn.Module = _KwargsAdapter()
        call_args: tuple = ()
    else:
        target = model
        call_args = args if args is not None else ()

    forward_flops = _profile_with_flop_counter(target, call_args)
    if forward_flops < 0:
        forward_flops = _profile_with_thop(target, call_args)

    return forward_flops * 3 if with_backward else forward_flops


def param_bytes(model: nn.Module, *, trainable_only: bool = False) -> int:
    """
    Total bytes occupied by model parameters (for MBU calculation).

    Args:
        trainable_only: If True, count only parameters that require grad.
                        Useful for PEFT/LoRA backward-pass MBU estimates.
    """
    params = (
        (p for p in model.parameters() if p.requires_grad)
        if trainable_only
        else model.parameters()
    )
    return sum(p.numel() * p.element_size() for p in params)
