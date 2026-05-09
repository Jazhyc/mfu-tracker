"""FLOP counting via FlopCounterMode (PyTorch 2.1+)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FlopProfile:
    """
    Two FLOP counts for one model step.

    ``flops`` is the unmasked matmul count reported by PyTorch's
    ``FlopCounterMode``. This is the convention used by the PaLM paper, the
    Chinchilla scaling laws, and most published MFU numbers.

    ``hfu_flops`` halves the SDPA contribution when ``is_causal=True`` to
    reflect what a causal flash-attention kernel actually executes (it skips
    the upper triangle of the attention matrix). Use this for Hardware FLOPs
    Utilization (HFU). For non-causal models ``hfu_flops == flops``.

    The two numbers come from a single trace.
    """
    flops: int
    hfu_flops: int


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


# Position of `is_causal` in each SDPA ATen op's positional args (0-indexed,
# excluding `out_shape` which the registry passes as a kwarg). Matches the
# schemas at:
#   aten._scaled_dot_product_flash_attention(q, k, v, dropout_p, is_causal, ...)
#   aten._scaled_dot_product_efficient_attention(q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, ...)
#   aten._scaled_dot_product_cudnn_attention(q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, ...)
_IS_CAUSAL_INDEX = {
    "_scaled_dot_product_flash_attention": 4,
    "_scaled_dot_product_efficient_attention": 6,
    "_scaled_dot_product_cudnn_attention": 6,
}


def _build_causal_aware_mapping(causal_savings_box: list[int]) -> dict[Any, Any]:
    """
    Custom flop-formula mapping that mirrors PyTorch's default sdpa_flop counts
    while accumulating the *causal savings* (= half the SDPA FLOPs when
    is_causal=True) into ``causal_savings_box[0]`` as a side channel.

    The reported total stays equal to the unhalved (PaLM/MFU) convention; HFU
    is recovered as ``total - causal_savings``.
    """
    from torch.utils.flop_counter import sdpa_flop_count

    aten = torch.ops.aten

    def _make_formula(is_causal_idx: int):
        def formula(query, key, value, *args, out_shape=None, **kwargs):
            full = sdpa_flop_count(query.shape, key.shape, value.shape)
            # is_causal lives among `args` — adjust for the q/k/v stripped above.
            extra_idx = is_causal_idx - 3
            is_causal = bool(args[extra_idx]) if extra_idx < len(args) else False
            if is_causal:
                causal_savings_box[0] += full // 2
            return full
        formula._get_raw = True  # type: ignore[attr-defined]
        return formula

    return {
        aten._scaled_dot_product_flash_attention: _make_formula(
            _IS_CAUSAL_INDEX["_scaled_dot_product_flash_attention"]
        ),
        aten._scaled_dot_product_efficient_attention: _make_formula(
            _IS_CAUSAL_INDEX["_scaled_dot_product_efficient_attention"]
        ),
        aten._scaled_dot_product_cudnn_attention: _make_formula(
            _IS_CAUSAL_INDEX["_scaled_dot_product_cudnn_attention"]
        ),
    }


def _profile_with_flop_counter(target: nn.Module, call_args: tuple) -> tuple[int, int]:
    """Profile using torch.utils.flop_counter.FlopCounterMode.

    Returns ``(forward_flops, causal_savings)`` — both non-negative, or ``(-1, 0)``
    on failure. ``causal_savings`` is the FLOP count that a causal flash-attention
    kernel skips relative to the unmasked matmul; subtract from ``forward_flops``
    for HFU.
    """
    from torch.utils.flop_counter import FlopCounterMode

    causal_savings_box = [0]
    custom_mapping = _build_causal_aware_mapping(causal_savings_box)

    was_training = target.training
    target.eval()
    try:
        fc = FlopCounterMode(display=False, custom_mapping=custom_mapping)
        with torch.no_grad(), fc:
            target(*call_args)
        return int(fc.get_total_flops()), causal_savings_box[0]
    except Exception:
        return -1, 0
    finally:
        target.train(was_training)


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

    **Quantized models (bitsandbytes INT8 / NF4):**
    Both counters operate on the Python/ATen graph and cannot see inside opaque
    bitsandbytes CUDA kernels. In practice the FLOPs reported are close to
    correct because NF4 (used by QLoRA) dequantizes weights to fp16 before the
    matmul, so the actual computation is a standard fp16 GEMM. INT8 similarly
    performs an fp16-equivalent matmul after dequantization in most bitsandbytes
    kernels. Pass the appropriate ``dtype`` to ``track()`` / ``compute_mfu()``
    to select the correct peak TFLOPS ceiling::

        # QLoRA (NF4 base + fp16 LoRA adapters) — adapters run in fp16
        flops = profile_flops(model, kwargs=batch, with_backward=False)
        with track(flops, param_bytes(model), dtype="fp16", spec=spec) as r:
            ...

        # Pure INT8 inference
        with track(flops, param_bytes(model), dtype="int8", spec=spec) as r:
            ...

    **PEFT / LoRA MBU:**
    ``param_bytes(model)`` counts all parameters including the frozen base,
    which is correct for the forward pass (all weights are read from memory).
    For a backward-pass MBU estimate that excludes frozen weights, use
    ``param_bytes(model, trainable_only=True)``::

        # Backward MBU for a LoRA-finetuned model
        active_bytes = param_bytes(model, trainable_only=True)

    Args:
        model:         The nn.Module to profile. Should be on CUDA for accurate
                       SDPA counts.
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

    forward_flops, _ = _profile_with_flop_counter(target, call_args)
    if forward_flops < 0:
        raise RuntimeError(
            "FlopCounterMode failed to trace this model. "
            "Compute FLOPs analytically (e.g. 6 × params × tokens) and pass them "
            "directly to compute_mfu()/track() instead of using profile_flops()."
        )

    return forward_flops * 3 if with_backward else forward_flops


def profile_flops_with_hfu(
    model: nn.Module,
    args: Optional[tuple] = None,
    kwargs: Optional[dict[str, Any]] = None,
    *,
    with_backward: bool = True,
) -> FlopProfile:
    """
    Like :func:`profile_flops` but returns both MFU- and HFU-style FLOP counts.

    ``FlopProfile.flops`` is the unmasked matmul total (PaLM convention, what
    most published MFU numbers use). ``FlopProfile.hfu_flops`` halves SDPA
    contributions for which ``is_causal=True`` — closer to what a causal
    flash-attention kernel actually executes.

    Both numbers come from one trace, so the cost is identical to
    :func:`profile_flops`. For non-causal models the two values are equal.

    Limitations of the HFU correction:
        * Only ``is_causal=True`` on `F.scaled_dot_product_attention` is
          detected. Models passing an explicit causal ``attn_mask`` instead of
          the flag are counted as non-causal.
        * Sliding-window / local-attention masks (e.g. Mistral, Gemma)
          actually skip more than half the matrix — halving understates the
          savings.
        * The math SDPA backend computes the full matrix even with
          ``is_causal=True``, so ``hfu_flops`` overstates the savings on CPU
          or when flash/efficient backends are unavailable.
        * Direct ``flash_attn_func`` calls bypass the ATen dispatch entirely;
          use :func:`flash_attn_flops` with ``causal=True`` to add the
          correction manually.
    """
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

    forward_flops, causal_savings = _profile_with_flop_counter(target, call_args)
    if forward_flops < 0:
        raise RuntimeError(
            "FlopCounterMode failed to trace this model. "
            "Compute FLOPs analytically (e.g. 6 × params × tokens) and pass them "
            "directly to compute_mfu()/track() instead of using profile_flops()."
        )

    forward_hfu = forward_flops - causal_savings
    if with_backward:
        return FlopProfile(flops=forward_flops * 3, hfu_flops=forward_hfu * 3)
    return FlopProfile(flops=forward_flops, hfu_flops=forward_hfu)


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
