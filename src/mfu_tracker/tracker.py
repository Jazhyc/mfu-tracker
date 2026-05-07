"""MFU and MBU measurement — context manager and standalone functions."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import torch

from .gpu import GPUSpec, get_gpu_spec


@dataclass
class UtilizationResult:
    """
    Result holder populated after a ``track()`` or ``track_step()`` block exits.

    Fields backed by CUDA events (from ``MFUOptimizerWrapper.track_step()``) are
    computed lazily on first access — the GPU sync is deferred until the value is
    actually needed. Fields from the CPU-timed ``track()`` context manager are
    populated eagerly since ``torch.cuda.synchronize`` is already called there.
    """

    dtype: str = "fp16"
    gpu_spec: Optional[GPUSpec] = None
    num_gpus: int = 1

    # Eagerly-set fields (track()) or lazily-set after _resolve() (track_step()).
    _mfu: Optional[float] = field(default=None, repr=False)
    _mbu: Optional[float] = field(default=None, repr=False)
    _elapsed_sec: Optional[float] = field(default=None, repr=False)
    _achieved_tflops: Optional[float] = field(default=None, repr=False)
    _achieved_tbs: Optional[float] = field(default=None, repr=False)

    # Set by MFUOptimizerWrapper for lazy resolution.
    _e_start: Any = field(default=None, repr=False)
    _e_end: Any = field(default=None, repr=False)
    _total_flops: Optional[int] = field(default=None, repr=False)
    _param_bytes: Optional[int] = field(default=None, repr=False)
    _device: Optional[torch.device] = field(default=None, repr=False)

    def _resolve(self) -> None:
        """Sync and compute all fields from CUDA events. Called at most once."""
        if self._e_start is None:
            return
        if self._total_flops is None or self._param_bytes is None:
            return
        assert self.gpu_spec is not None  # set together with _e_start
        torch.cuda.synchronize(self._device)
        elapsed = self._e_start.elapsed_time(self._e_end) / 1000

        peak_tflops = self.gpu_spec.peak_tflops(self.dtype) * self.num_gpus
        peak_tbs = self.gpu_spec.peak_memory_bandwidth_tbs * self.num_gpus

        achieved_tflops = self._total_flops / elapsed / 1e12
        achieved_tbs = self._param_bytes / elapsed / 1e12

        self._elapsed_sec = elapsed
        self._achieved_tflops = achieved_tflops
        self._achieved_tbs = achieved_tbs
        self._mfu = achieved_tflops / peak_tflops
        self._mbu = achieved_tbs / peak_tbs
        self._e_start = None  # mark resolved

    @property
    def mfu(self) -> Optional[float]:
        self._resolve()
        return self._mfu

    @mfu.setter
    def mfu(self, v: Optional[float]) -> None:
        self._mfu = v

    @property
    def mbu(self) -> Optional[float]:
        self._resolve()
        return self._mbu

    @mbu.setter
    def mbu(self, v: Optional[float]) -> None:
        self._mbu = v

    @property
    def elapsed_sec(self) -> Optional[float]:
        self._resolve()
        return self._elapsed_sec

    @elapsed_sec.setter
    def elapsed_sec(self, v: Optional[float]) -> None:
        self._elapsed_sec = v

    @property
    def achieved_tflops(self) -> Optional[float]:
        self._resolve()
        return self._achieved_tflops

    @achieved_tflops.setter
    def achieved_tflops(self, v: Optional[float]) -> None:
        self._achieved_tflops = v

    @property
    def achieved_tbs(self) -> Optional[float]:
        self._resolve()
        return self._achieved_tbs

    @achieved_tbs.setter
    def achieved_tbs(self, v: Optional[float]) -> None:
        self._achieved_tbs = v


@contextmanager
def track(
    flop_count: int,
    param_bytes: int,
    *,
    dtype: str = "fp16",
    num_gpus: int = 1,
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> Generator[UtilizationResult, None, None]:
    """
    Context manager that measures MFU and MBU for an arbitrary compute block.

    Args:
        flop_count:  Total FLOPs for the block (use flops.profile_flops or your own).
        param_bytes: Bytes transferred for weights (num_params * bytes_per_element).
        dtype:       Compute dtype string — "fp16", "bf16", "int8", "fp8", "int4", "fp4".
        num_gpus:    Multiplier applied to the peak ceiling (default 1). When using
                     ``profile_flops`` as the source of *flop_count*, leave this at 1
                     regardless of parallelism strategy — ``profile_flops`` returns
                     per-GPU FLOPs, and per-GPU MFU equals global MFU for all standard
                     parallelism types (the N factors cancel). Only set this when
                     *flop_count* is the analytically-derived *full-model* FLOP count
                     (e.g. ``6 × params × tokens``) rather than a profiled per-GPU count.
        device:      CUDA device to measure against (default: current device).
        spec:        Pre-queried GPUSpec; fetched once if not provided.

    Yields a :class:`UtilizationResult` whose fields are ``None`` until the block
    exits, then populated with measured values.

    Example — single GPU::

        flops = profile_flops(model, args=(sample,), with_backward=True)
        with track(flops, param_bytes(model), dtype="bf16") as result:
            loss = model(inputs)
            loss.backward()
            optimizer.step()
        print(f"MFU={result.mfu:.1%}  MBU={result.mbu:.1%}")

    Example — any parallelism strategy (DDP, FSDP, tensor, pipeline)::

        # profile_flops on whatever model shard the local rank holds gives
        # per-GPU FLOPs; per-GPU MFU == global MFU for all standard strategies.
        with track(flops, param_bytes(model), dtype="bf16") as r:
            ...
    """
    if spec is None:
        spec = get_gpu_spec(device)

    peak_tflops = spec.peak_tflops(dtype) * num_gpus
    peak_tbs = spec.peak_memory_bandwidth_tbs * num_gpus

    result = UtilizationResult(dtype=dtype, gpu_spec=spec, num_gpus=num_gpus)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    yield result
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    result.elapsed_sec = elapsed
    result.achieved_tflops = flop_count / elapsed / 1e12
    result.achieved_tbs = param_bytes / elapsed / 1e12
    result.mfu = result.achieved_tflops / peak_tflops
    result.mbu = result.achieved_tbs / peak_tbs


def compute_mfu(
    flop_count: int,
    elapsed_sec: float,
    *,
    dtype: str = "fp16",
    num_gpus: int = 1,
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> float:
    """
    Standalone MFU calculation without a context manager.

    Args:
        num_gpus: GPUs in the peak ceiling. See :func:`track` for guidance.
    """
    if spec is None:
        spec = get_gpu_spec(device)
    return (flop_count / elapsed_sec / 1e12) / (spec.peak_tflops(dtype) * num_gpus)


def compute_mbu(
    param_bytes: int,
    elapsed_sec: float,
    *,
    num_gpus: int = 1,
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> float:
    """
    Standalone MBU calculation without a context manager.

    Args:
        num_gpus: GPUs in the peak ceiling. See :func:`track` for guidance.
    """
    if spec is None:
        spec = get_gpu_spec(device)
    return (param_bytes / elapsed_sec / 1e12) / (spec.peak_memory_bandwidth_tbs * num_gpus)
