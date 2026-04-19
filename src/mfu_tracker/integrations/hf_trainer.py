"""HuggingFace Trainer integration via TrainerCallback."""
from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from ..flops import param_bytes, profile_flops
from ..gpu import GPUSpec, get_gpu_spec
from ..tracker import compute_mbu, compute_mfu


class MFUCallback:
    """
    TrainerCallback that logs MFU and MBU at every Trainer logging step.

    We do NOT read Trainer's ``state.total_flos`` because HF uses the dense-model
    6ND formula for all architectures — overcounting MoE models by up to 4×.
    Instead we profile the model once with calflops (which instruments actual ops),
    cache the per-step FLOP count, then divide by measured wall time.

    Usage::

        from mfu_tracker.integrations.hf_trainer import MFUCallback

        callback = MFUCallback(sample_batch=next(iter(train_dataloader)), dtype="bf16")
        trainer = Trainer(..., callbacks=[callback])

    Args:
        sample_batch: A representative batch dict (passed as ``**kwargs`` to the model).
                      Used once at training start to profile FLOPs. Must match the
                      shape of real training batches.
        dtype:        Compute dtype string for the peak ceiling — "fp16", "bf16",
                      "int8", "fp8", etc.
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

        self._spec: Optional[GPUSpec] = None
        self._flops_per_step: Optional[int] = None
        self._param_bytes: Optional[int] = None

        self._step_start: Optional[float] = None
        self._acc_time: float = 0.0
        self._acc_steps: int = 0

    def _profile(self, model: nn.Module) -> None:
        self._spec = get_gpu_spec(self.device)
        self._param_bytes = param_bytes(model)
        try:
            self._flops_per_step = profile_flops(
                model,
                kwargs=self.sample_batch,
                with_backward=True,
            )
        except Exception as exc:
            warnings.warn(
                f"mfu-tracker: calflops profiling failed ({exc}); MFU will not be logged.",
                stacklevel=2,
            )

    # --- TrainerCallback protocol -------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._profile(model)

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize(self.device)
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is None:
            return
        torch.cuda.synchronize(self.device)
        self._acc_time += time.perf_counter() - self._step_start
        self._acc_steps += 1
        self._step_start = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._flops_per_step is None or self._acc_time == 0:
            return

        total_flops = self._flops_per_step * self._acc_steps
        total_bytes = self._param_bytes * self._acc_steps

        logs["mfu"] = round(
            compute_mfu(total_flops, self._acc_time, dtype=self.dtype, spec=self._spec), 4
        )
        logs["mbu"] = round(
            compute_mbu(total_bytes, self._acc_time, spec=self._spec), 4
        )

        self._acc_time = 0.0
        self._acc_steps = 0
