"""MFU and MBU measurement — context manager and standalone functions."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch

from .gpu import GPUSpec, get_gpu_spec


@dataclass
class UtilizationResult:
    """Mutable result holder — fields are populated after the context block exits."""
    mfu: Optional[float] = None
    mbu: Optional[float] = None
    elapsed_sec: Optional[float] = None
    achieved_tflops: Optional[float] = None
    achieved_tbs: Optional[float] = None
    dtype: str = "fp16"
    gpu_spec: Optional[GPUSpec] = None


@contextmanager
def track(
    flop_count: int,
    param_bytes: int,
    *,
    dtype: str = "fp16",
    device: Optional[torch.device] = None,
    spec: Optional[GPUSpec] = None,
) -> Generator[UtilizationResult, None, None]:
    """
    Context manager that measures MFU and MBU for an arbitrary compute block.

    Args:
        flop_count:  Total FLOPs for the block (use flops.profile_flops or your own).
        param_bytes: Bytes transferred for weights (num_params * bytes_per_element).
        dtype:       Compute dtype string — "fp16", "bf16", "int8", "fp8", "int4", "fp4".
        device:      CUDA device to measure against (default: current device).
        spec:        Pre-queried GPUSpec; fetched once if not provided.

    Yields a :class:`UtilizationResult` whose fields are ``None`` until the block
    exits, then populated with the measured values.

    Example::

        flops = profile_flops(model, args=(sample,), with_backward=True)
        with track(flops, param_bytes(model), dtype="bf16") as result:
            loss = model(inputs)
            loss.backward()
            optimizer.step()
        print(f"MFU={result.mfu:.1%}  MBU={result.mbu:.1%}")
    """
    if spec is None:
        spec = get_gpu_spec(device)

    result = UtilizationResult(dtype=dtype, gpu_spec=spec)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    yield result
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    result.elapsed_sec = elapsed
    result.achieved_tflops = flop_count / elapsed / 1e12
    result.achieved_tbs = param_bytes / elapsed / 1e12
    result.mfu = result.achieved_tflops / spec.peak_tflops(dtype)
    result.mbu = result.achieved_tbs / spec.peak_memory_bandwidth_tbs


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
