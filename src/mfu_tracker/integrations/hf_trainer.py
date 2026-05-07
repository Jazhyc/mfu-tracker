"""HuggingFace Trainer integration via TrainerCallback."""
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import TrainerCallback

from ..flops import param_bytes, profile_flops
from ..gpu import GPUSpec, get_gpu_spec
from ..tracker import compute_mbu, compute_mfu


class MFUCallback(TrainerCallback):
    """
    TrainerCallback that logs MFU and MBU at every Trainer logging step.

    FLOPs per step are ``fwd_flops × (1 + backward_factor)`` where
    ``backward_factor`` defaults to 2.0 (standard 3× convention). Set it higher
    when using gradient checkpointing (typical: 3.0–4.0).

    Per-step cost is two non-blocking ``Event.record()`` calls (~10 μs CPU, no
    GPU stall). The single ``torch.cuda.synchronize()`` is deferred to ``on_log``
    and amortised across all steps in the logging interval.

    Usage::

        from mfu_tracker.integrations.hf_trainer import MFUCallback

        callback = MFUCallback(sample_batch=next(iter(train_dataloader)), dtype="bf16")
        # num_gpus is auto-detected from torch.distributed — no manual config needed
        trainer = Trainer(..., callbacks=[callback])

    **torch.compile**: profile_flops is called at ``on_train_begin``, before the
    first compiled step. This is correct — ``torch.compile`` does not change the
    FLOP count (same math, just faster execution). The MFU improvement from
    compilation is captured automatically in the CUDA event timing of real steps.
    Do not pass a compiled model to this callback directly; let HF Trainer compile
    after the callback is registered.

    **DDP / FSDP**: leave ``num_gpus=1`` — per-GPU MFU equals global MFU for
    data-parallel jobs.

    Args:
        sample_batch:    A representative batch dict passed as ``**kwargs`` to the
                         model. Used once at training start to profile forward FLOPs.
        dtype:           Compute dtype for the peak ceiling — "fp16", "bf16", etc.
        num_gpus:        GPUs in the peak ceiling (default 1).
        backward_factor: Multiplier for backward pass cost (default 2.0). Set
                         higher when using gradient checkpointing (typical: 3.0–4.0).
        metric_prefix:   Prefix for logged metric names (default ``"throughput"``).
                         Results in ``throughput/mfu`` and ``throughput/mbu``, which
                         WandB groups into its own section away from loss/lr. Set to
                         ``""`` to log bare ``mfu`` / ``mbu`` keys.
        device:          CUDA device to query. Defaults to current device.
    """

    def __init__(
        self,
        sample_batch: dict[str, Any],
        dtype: str = "bf16",
        num_gpus: int = 1,
        backward_factor: float = 2.0,
        metric_prefix: str = "throughput",
        device: Optional[torch.device] = None,
    ) -> None:
        self.sample_batch = sample_batch
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.backward_factor = backward_factor
        self.metric_prefix = metric_prefix
        self.device = device

        self._model: Optional[nn.Module] = None
        self._spec: Optional[GPUSpec] = None
        self._fwd_flops: Optional[int] = None
        self._param_bytes: Optional[int] = None

        # Each entry: (e_start, e_bwd, e_end, bwd_recorded)
        # Each entry: (e_start, e_end). Accumulated between on_log calls.
        self._pending: list[tuple] = []

    def _profile(self, model: nn.Module) -> None:
        self._spec = get_gpu_spec(self.device)
        self._param_bytes = param_bytes(model)
        try:
            # Move sample batch to the model's device (Trainer may have moved the model).
            model_device = next(model.parameters()).device
            batch = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in self.sample_batch.items()
            }
            self._fwd_flops = profile_flops(
                model, kwargs=batch, with_backward=False
            )
        except Exception as exc:
            warnings.warn(
                f"mfu-tracker: FLOP profiling failed ({exc}); MFU will not be logged.",
                stacklevel=2,
            )

    # --- TrainerCallback protocol -------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and torch.cuda.is_available():
            self._model = model
            self._profile(model)

    def on_step_begin(self, args, state, control, **kwargs):
        if self._fwd_flops is None or not torch.cuda.is_available():
            return

        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        self._pending_step = (e_start, e_end)

    def on_step_end(self, args, state, control, **kwargs):
        if not hasattr(self, "_pending_step"):
            return

        e_start, e_end = self._pending_step
        e_end.record()
        self._pending.append((e_start, e_end))
        del self._pending_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self._pending or self._fwd_flops is None or self._param_bytes is None:
            return

        # Single sync amortised across all accumulated steps.
        torch.cuda.synchronize(self.device)

        n_steps = len(self._pending)
        total_ms = sum(e_start.elapsed_time(e_end) for e_start, e_end in self._pending)
        self._pending.clear()

        elapsed_sec = total_ms / 1000
        if elapsed_sec <= 0:
            return

        total_flops = int(self._fwd_flops * n_steps * (1 + self.backward_factor))

        prefix = f"{self.metric_prefix}/" if self.metric_prefix else ""
        logs[f"{prefix}mfu"] = round(
            compute_mfu(total_flops, elapsed_sec, dtype=self.dtype, num_gpus=self.num_gpus, spec=self._spec), 4
        )
        logs[f"{prefix}mbu"] = round(
            compute_mbu(self._param_bytes * n_steps, elapsed_sec, num_gpus=self.num_gpus, spec=self._spec), 4
        )
