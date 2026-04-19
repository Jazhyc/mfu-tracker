"""MFU and MBU measurement — context manager and standalone functions."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import torch

from .flops import transformer_flops  # noqa: F401 – re-exported for convenience
from .gpu import GPUSpec, get_gpu_spec


@dataclass(frozen=True)
class UtilizationResult:
    mfu: float                  # Model FLOPs Utilization [0, 1]
    mbu: float                  # Model Bandwidth Utilization [0, 1]
    elapsed_sec: float
    achieved_tflops: float
    achieved_tbs: float         # TB/s
    dtype: str
    gpu_spec: GPUSpec


@contextmanager
def track(
    flop_count: int,
    param_bytes: int,
    *,
    dtype: str = "fp16",
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> Generator[None, None, UtilizationResult]:
    """
    Context manager that measures MFU and MBU for an arbitrary compute block.

    Args:
        flop_count:  Total FLOPs for the block (use flops.transformer_flops or your own).
        param_bytes: Bytes transferred for weights (num_params * bytes_per_element).
        dtype:       Compute dtype string — "fp16", "bf16", "int8", "fp8", "int4", "fp4".
                     Selects the correct hardware peak ceiling for MFU.
        device:      CUDA device to measure against (default: current device).
        spec:        Pre-queried GPUSpec; fetched once if not provided.

    Yields nothing; the return value is available as the `as` target after the block.

    Example::

        flops = transformer_flops(num_params, batch * seq_len)
        with track(flops, param_bytes, dtype="bf16") as result:
            model(inputs)
        print(f"MFU={result.mfu:.1%}  MBU={result.mbu:.1%}")
    """
    if spec is None:
        spec = get_gpu_spec(device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    achieved_tflops = flop_count / elapsed / 1e12
    achieved_tbs = param_bytes / elapsed / 1e12

    mfu = achieved_tflops / spec.peak_tflops(dtype)
    mbu = achieved_tbs / spec.peak_memory_bandwidth_tbs

    return UtilizationResult(
        mfu=mfu,
        mbu=mbu,
        elapsed_sec=elapsed,
        achieved_tflops=achieved_tflops,
        achieved_tbs=achieved_tbs,
        dtype=dtype,
        gpu_spec=spec,
    )


def compute_mfu(
    flop_count: int,
    elapsed_sec: float,
    *,
    dtype: str = "fp16",
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> float:
    """Standalone MFU calculation without a context manager."""
    if spec is None:
        spec = get_gpu_spec(device)
    return (flop_count / elapsed_sec / 1e12) / spec.peak_tflops(dtype)


def compute_mbu(
    param_bytes: int,
    elapsed_sec: float,
    *,
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> float:
    """Standalone MBU calculation without a context manager."""
    if spec is None:
        spec = get_gpu_spec(device)
    return (param_bytes / elapsed_sec / 1e12) / spec.peak_memory_bandwidth_tbs
