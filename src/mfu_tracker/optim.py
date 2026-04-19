"""Optimizer wrapper that dynamically measures MFU and MBU per training step."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch
import torch.nn as nn

from .flops import param_bytes, profile_flops
from .gpu import GPUSpec, get_gpu_spec
from .tracker import UtilizationResult, compute_mbu, compute_mfu


class MFUOptimizerWrapper:
    """
    Wraps any ``torch.optim.Optimizer`` to automatically track MFU and MBU.

    The backward-pass cost is measured dynamically via CUDA events and a gradient
    hook — no need to specify a backward factor or detect gradient checkpointing
    manually. The ratio is derived from the actual wall time split between forward
    and backward each step.

    ``zero_grad()`` is called automatically inside ``track_step()``, so it should
    be removed from the training loop.

    Usage::

        optimizer = MFUOptimizerWrapper(
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            model=model,
            sample_batch=sample_batch,
            dtype="bf16",
        )

        for batch in dataloader:
            with optimizer.track_step() as result:
                output = model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
            print(f"MFU={result.mfu:.1%}  MBU={result.mbu:.1%}")
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        sample_batch: dict[str, Any],
        dtype: str = "bf16",
        device: Optional[torch.device] = None,
    ) -> None:
        self.optimizer = optimizer
        self._model = model
        self._sample_batch = sample_batch
        self._dtype = dtype
        self._device = device

        self._spec: Optional[GPUSpec] = None
        self._fwd_flops: Optional[int] = None
        self._param_bytes: Optional[int] = None

    def _profile_once(self) -> None:
        self._spec = get_gpu_spec(self._device)
        try:
            self._fwd_flops = profile_flops(
                self._model,
                kwargs=self._sample_batch,
                with_backward=False,
            )
        except Exception as exc:
            warnings.warn(
                f"mfu-tracker: profiling failed ({exc}); MFU will not be populated.",
                stacklevel=3,
            )
        self._param_bytes = param_bytes(self._model)

    @contextmanager
    def track_step(self) -> Generator[UtilizationResult, None, None]:
        """
        Context manager that wraps one training step and populates a
        :class:`~mfu_tracker.UtilizationResult` with MFU and MBU.

        ``optimizer.zero_grad()`` is called automatically when the block exits.
        The backward factor (ratio of backward to forward time) is measured
        dynamically via CUDA events and a gradient hook, so gradient checkpointing
        and other effects are captured without any manual configuration.
        """
        if self._spec is None:
            self._profile_once()

        e_start = torch.cuda.Event(enable_timing=True)
        e_bwd = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        bwd_recorded = [False]

        def _bwd_hook(grad: torch.Tensor) -> torch.Tensor:
            if not bwd_recorded[0]:
                e_bwd.record()
                bwd_recorded[0] = True
            return grad

        # Hook the last trainable parameter — it receives its gradient first
        # during backward (parameters are traversed in reverse forward order).
        handle = None
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        if trainable:
            handle = trainable[-1].register_hook(_bwd_hook)

        result = UtilizationResult(dtype=self._dtype, gpu_spec=self._spec)

        e_start.record()
        try:
            yield result
            self.optimizer.zero_grad()
        finally:
            e_end.record()
            if handle is not None:
                handle.remove()

            torch.cuda.synchronize(self._device)

            total_ms = e_start.elapsed_time(e_end)
            elapsed_sec = total_ms / 1000

            if bwd_recorded[0] and self._fwd_flops is not None:
                fwd_ms = e_start.elapsed_time(e_bwd)
                bwd_ms = total_ms - fwd_ms
                backward_factor = bwd_ms / fwd_ms if fwd_ms > 0 else 2.0
                total_flops = int(self._fwd_flops * (1 + backward_factor))
            elif self._fwd_flops is not None:
                # No backward was run (inference-only step).
                total_flops = self._fwd_flops
            else:
                return

            result.elapsed_sec = elapsed_sec
            result.achieved_tflops = total_flops / elapsed_sec / 1e12
            result.achieved_tbs = self._param_bytes / elapsed_sec / 1e12
            result.mfu = compute_mfu(
                total_flops, elapsed_sec, dtype=self._dtype, spec=self._spec
            )
            result.mbu = compute_mbu(
                self._param_bytes, elapsed_sec, spec=self._spec
            )

    # --- Proxy the underlying optimizer ------------------------------------

    def step(self, *args, **kwargs) -> None:
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)
