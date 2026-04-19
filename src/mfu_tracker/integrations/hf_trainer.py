"""HuggingFace Trainer integration via TrainerCallback."""
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from ..flops import param_bytes, profile_flops
from ..gpu import GPUSpec, get_gpu_spec
from ..tracker import UtilizationResult, compute_mbu, compute_mfu


class MFUCallback:
    """
    TrainerCallback that logs MFU and MBU at every Trainer logging step.

    Uses the same gradient hook + CUDA event technique as ``MFUOptimizerWrapper``
    to measure the forward/backward split dynamically — gradient checkpointing and
    recomputation are captured automatically without any configuration.

    Per-step cost is three non-blocking ``Event.record()`` calls and one Python
    gradient hook dispatch (~10–50 μs CPU, no GPU stall). The single
    ``torch.cuda.synchronize()`` is deferred to ``on_log`` and amortised across
    all steps in the logging interval.

    Usage::

        from mfu_tracker.integrations.hf_trainer import MFUCallback

        callback = MFUCallback(sample_batch=next(iter(train_dataloader)), dtype="bf16")
        trainer = Trainer(..., callbacks=[callback])

    Args:
        sample_batch: A representative batch dict passed as ``**kwargs`` to the model.
                      Used once at training start to profile forward FLOPs. Shape must
                      match real training batches.
        dtype:        Compute dtype for the peak ceiling — "fp16", "bf16", "int8", etc.
        device:       CUDA device to query. Defaults to current device.
    """

    def __init__(
        self,
        sample_batch: dict[str, Any],
        dtype: str = "bf16",
        device: Optional[torch.device] = None,
    ) -> None:
        self.sample_batch = sample_batch
        self.dtype = dtype
        self.device = device

        self._model: Optional[nn.Module] = None
        self._spec: Optional[GPUSpec] = None
        self._fwd_flops: Optional[int] = None
        self._param_bytes: Optional[int] = None

        # Each entry: (e_start, e_bwd, e_end, bwd_recorded)
        # Accumulated between on_log calls; flushed on each on_log.
        self._pending: list[tuple] = []
        self._current_hook_handle = None

    def _profile(self, model: nn.Module) -> None:
        self._spec = get_gpu_spec(self.device)
        self._param_bytes = param_bytes(model)
        try:
            self._fwd_flops = profile_flops(
                model, kwargs=self.sample_batch, with_backward=False
            )
        except Exception as exc:
            warnings.warn(
                f"mfu-tracker: FLOP profiling failed ({exc}); MFU will not be logged.",
                stacklevel=2,
            )

    # --- TrainerCallback protocol -------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._model = model
            self._profile(model)

    def on_step_begin(self, args, state, control, **kwargs):
        if self._fwd_flops is None or self._model is None:
            return

        e_start = torch.cuda.Event(enable_timing=True)
        e_bwd = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        bwd_recorded = [False]

        def _bwd_hook(grad: torch.Tensor) -> torch.Tensor:
            if not bwd_recorded[0]:
                e_bwd.record()
                bwd_recorded[0] = True
            return grad

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        if trainable:
            self._current_hook_handle = trainable[-1].register_hook(_bwd_hook)

        e_start.record()
        # Stash references so on_step_end can close the interval and read bwd_recorded.
        self._pending_step = (e_start, e_bwd, e_end, bwd_recorded)

    def on_step_end(self, args, state, control, **kwargs):
        if not hasattr(self, "_pending_step"):
            return

        e_start, e_bwd, e_end, bwd_recorded = self._pending_step
        e_end.record()

        if self._current_hook_handle is not None:
            self._current_hook_handle.remove()
            self._current_hook_handle = None

        self._pending.append((e_start, e_bwd, e_end, bwd_recorded))
        del self._pending_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self._pending or self._fwd_flops is None:
            return

        # Single sync amortised across all accumulated steps.
        torch.cuda.synchronize(self.device)

        n_steps = len(self._pending)
        total_ms = 0.0
        fwd_ms = 0.0
        n_with_bwd = 0

        for e_start, e_bwd, e_end, bwd_recorded in self._pending:
            total_ms += e_start.elapsed_time(e_end)
            if bwd_recorded[0]:
                fwd_ms += e_start.elapsed_time(e_bwd)
                n_with_bwd += 1

        self._pending.clear()

        elapsed_sec = total_ms / 1000
        if elapsed_sec <= 0:
            return

        if n_with_bwd > 0:
            backward_factor = (total_ms - fwd_ms) / fwd_ms if fwd_ms > 0 else 2.0
        else:
            backward_factor = 2.0

        total_flops = int(self._fwd_flops * n_steps * (1 + backward_factor))

        logs["mfu"] = round(
            compute_mfu(total_flops, elapsed_sec, dtype=self.dtype, spec=self._spec), 4
        )
        logs["mbu"] = round(
            compute_mbu(self._param_bytes * n_steps, elapsed_sec, spec=self._spec), 4
        )
