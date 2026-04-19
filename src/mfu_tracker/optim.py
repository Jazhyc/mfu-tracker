"""Optimizer wrapper that dynamically measures MFU and MBU per training step."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch
import torch.nn as nn

from .flops import param_bytes, profile_flops
from .gpu import GPUSpec, get_gpu_spec
from .tracker import UtilizationResult


class MFUOptimizerWrapper:
    """
    Wraps any ``torch.optim.Optimizer`` to automatically track MFU and MBU.

    The backward-pass cost is measured dynamically via CUDA events and a gradient
    hook — no need to specify a backward factor or detect gradient checkpointing
    manually. The ratio is derived from the actual wall time split between forward
    and backward each step.

    ``zero_grad()`` is called automatically at the *start* of ``track_step()``.
    Call ``optimizer.step()`` **after** the block so it is excluded from the
    timing window (otherwise optimizer time inflates the apparent backward factor).

    Usage::

        optimizer = MFUOptimizerWrapper(
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            model=model,
            sample_batch=sample_batch,
            dtype="bf16",
            # num_gpus auto-detected from torch.distributed — omit unless overriding
        )

    **torch.compile**: pass the *uncompiled* model and compile separately after
    constructing the wrapper. ``profile_flops`` is called lazily on the first
    ``track_step()`` — if the model is already compiled at that point,
    ``FlopCounterMode`` may not count correctly. Compile after wrapping::

        optimizer = MFUOptimizerWrapper(raw_model, ...)
        compiled_model = torch.compile(raw_model)

        for batch in dataloader:
            with optimizer.track_step() as result:
                output = model(**batch)
                loss = output.loss
                loss.backward()
            optimizer.step()   # outside block — excluded from MFU timing
            print(f"MFU={result.mfu:.1%}  MBU={result.mbu:.1%}")
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        sample_batch: dict[str, Any],
        dtype: str = "bf16",
        num_gpus: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.optimizer = optimizer
        self._model = model
        self._sample_batch = sample_batch
        self._dtype = dtype
        self._num_gpus = num_gpus
        self._device = device

        self._spec: Optional[GPUSpec] = None
        self._fwd_flops: Optional[int] = None
        self._param_bytes: Optional[int] = None

    def profile(self) -> None:
        """
        Explicitly profile FLOPs on the current (uncompiled) model.

        Call this before ``torch.compile`` so the FLOP count is measured on the
        original graph. If not called, profiling happens lazily on the first
        ``track_step()`` — which may be too late if the model is already compiled::

            optimizer = MFUOptimizerWrapper(raw_optimizer, model, sample_batch, dtype="bf16")
            optimizer.profile()          # profile uncompiled model
            model = torch.compile(model) # compile after profiling
        """
        if self._spec is None:
            self._profile_once()

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

        ``optimizer.zero_grad()`` is called automatically at the *start* of the
        block so that ``optimizer.step()`` can be called **after** the block
        without corrupting gradient state. Timing captures forward + backward
        only; call ``step()`` outside the block for accurate MFU::

            with wrapped.track_step() as result:
                out = model(**batch)
                out.loss.backward()
            wrapped.step()   # outside — excluded from timing

        The backward factor (ratio of backward to forward time) is measured
        dynamically via CUDA events and a gradient hook, so gradient checkpointing
        and other effects are captured without any manual configuration.
        """
        if self._spec is None:
            self._profile_once()

        self.optimizer.zero_grad()

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

        result = UtilizationResult(dtype=self._dtype, gpu_spec=self._spec, num_gpus=self._num_gpus)

        e_start.record()
        try:
            yield result
        finally:
            e_end.record()
            if handle is not None:
                handle.remove()

            # Store events on the result for lazy resolution — no sync here.
            result._e_start = e_start
            result._e_bwd = e_bwd
            result._e_end = e_end
            result._bwd_recorded = bwd_recorded[0]
            result._fwd_flops = self._fwd_flops
            result._param_bytes = self._param_bytes
            result._device = self._device

    # --- Proxy the underlying optimizer ------------------------------------

    def step(self, *args, **kwargs) -> None:
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)
