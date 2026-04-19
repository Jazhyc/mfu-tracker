"""Optimizer wrapper that measures MFU and MBU per training step."""
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

    FLOPs are profiled once on the uncompiled model and scaled by
    ``1 + backward_factor`` (default 2.0, the standard forward + 2× backward
    convention). Set ``backward_factor`` higher when using gradient checkpointing,
    which recomputes activations during backward (typical values: 3.0–4.0).

    ``zero_grad()`` is called automatically at the *start* of ``track_step()``.
    Call ``optimizer.step()`` **after** the block so it is excluded from the
    timing window::

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

    **torch.compile**: profile the uncompiled model first, then compile::

        optimizer = MFUOptimizerWrapper(raw_model, ...)
        optimizer.profile()          # profile before compile
        model = torch.compile(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        sample_batch: dict[str, Any],
        dtype: str = "bf16",
        num_gpus: int = 1,
        backward_factor: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.optimizer = optimizer
        self._model = model
        self._sample_batch = sample_batch
        self._dtype = dtype
        self._num_gpus = num_gpus
        self._backward_factor = backward_factor
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
        block. Call ``optimizer.step()`` **after** the block so it is excluded
        from the timing window::

            with wrapped.track_step() as result:
                out = model(**batch)
                out.loss.backward()
            wrapped.step()

        FLOPs are ``fwd_flops × (1 + backward_factor)`` where ``backward_factor``
        defaults to 2.0 (standard 3× convention). Set it higher when using
        gradient checkpointing (typical: 3.0–4.0).
        """
        if self._spec is None:
            self._profile_once()

        self.optimizer.zero_grad()

        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        result = UtilizationResult(dtype=self._dtype, gpu_spec=self._spec, num_gpus=self._num_gpus)

        e_start.record()
        try:
            yield result
        finally:
            e_end.record()
            result._e_start = e_start
            result._e_end = e_end
            result._total_flops = (
                int(self._fwd_flops * (1 + self._backward_factor))
                if self._fwd_flops is not None else None
            )
            result._param_bytes = self._param_bytes
            result._device = self._device

    # --- Proxy the underlying optimizer ------------------------------------

    def step(self, *args, **kwargs) -> None:
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)
