"""FLOP counting via thop — works for any nn.Module architecture."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def profile_flops(
    model: nn.Module,
    args: Optional[tuple] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    with_backward: bool = True,
) -> int:
    """
    Count FLOPs for one forward (or forward+backward) pass through *model*.

    Uses thop to instrument actual torch ops via hooks, so it correctly
    handles MoE sparse routing, CNNs, RNNs, and any custom nn.Module —
    not just dense transformers.

    Args:
        model:         The nn.Module to profile.
        args:          Positional inputs passed to model(*args, **kwargs).
        kwargs:        Keyword inputs passed to model(*args, **kwargs).
        with_backward: Include backward-pass FLOPs (default True for training).
                       Backward pass is estimated as 2× forward FLOPs, giving
                       3× forward total — matching the standard 6ND convention.

    Returns:
        Total integer FLOP count for one step.
    """
    from thop import profile

    # thop calls model(*inputs), so kwargs need to be baked in via a wrapper.
    if kwargs:
        _kw = kwargs

        class _KwargsAdapter(nn.Module):
            def forward(self):
                return model(**(args[0] if args and isinstance(args[0], dict) else _kw))

        macs, _ = profile(_KwargsAdapter(), inputs=(), verbose=False)
    else:
        inputs = args if args is not None else ()
        macs, _ = profile(model, inputs=inputs, verbose=False)
    # thop returns MACs; 1 MAC = 2 FLOPs (multiply + accumulate)
    forward_flops = int(macs * 2)
    return forward_flops * 3 if with_backward else forward_flops


def param_bytes(model: nn.Module) -> int:
    """Total bytes occupied by all model parameters (for MBU calculation)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
