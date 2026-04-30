"""GPU capability querying and peak throughput derivation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch

# FP16 dense tensor-core FLOPs per SM per clock cycle.
# Keyed by (major, minor) compute capability.
# Values empirically validated against NVIDIA spec-sheet peaks:
#   A100 SXM4 : 156 TFLOPS = 108 SMs × 1024 × 1.410 GHz ✓
#   H100 SXM5 : 989 TFLOPS = 132 SMs × 3840 × 1.980 GHz ✓
#   RTX 4090  : ~660 TFLOPS = 128 SMs × 2048 × 2.520 GHz ✓
#
# Data-centre vs. consumer Ampere (8.0 vs. 8.6) differ because GA100's
# tensor cores are physically larger than GA102's.
_FP16_FLOPS_PER_SM_PER_CLOCK: dict[tuple[int, int], int] = {
    (7, 0): 1024,   # Volta         (V100)
    (7, 5): 1024,   # Turing        (T4, RTX 20xx)
    (8, 0): 1024,   # Ampere DC     (A100, A30)
    (8, 6): 512,    # Ampere cons.  (A10, RTX 30xx)
    (8, 7): 512,    # Ampere Jetson (Orin)
    (8, 9): 2048,   # Ada Lovelace  (RTX 40xx, L40S)
    (9, 0): 3840,   # Hopper        (H100, H200)
    (10, 0): 7680,  # Blackwell     (B100, B200) — estimated ≈ 2× Hopper
}

# Throughput multiplier relative to FP16 dense for each dtype.
# Only available from certain compute capabilities onward.
# (cc_major_min, multiplier)
_DTYPE_MULTIPLIER: dict[str, tuple[int, float]] = {
    "fp16":  (7, 1.0),
    "bf16":  (8, 1.0),   # BF16 tensor cores added in Ampere
    "int8":  (7, 2.0),   # INT8 tensor cores since Turing
    "fp8":   (9, 2.0),   # FP8 via Transformer Engine; Ada (8.9) also supports it
    "int4":  (7, 4.0),   # INT4 since Turing
    "fp4":   (10, 4.0),  # FP4 native in Blackwell
}
# Ada also supports FP8 even though its major is 8
_ADA_FP8_MINOR = 9


@dataclass(frozen=True)
class GPUSpec:
    name: str
    compute_capability: tuple[int, int]
    num_sms: int
    clock_rate_hz: float
    memory_bandwidth_bytes_per_sec: float
    peak_flops_by_dtype: dict[str, float] = field(default_factory=dict)
    peak_memory_bandwidth_tbs: float = 0.0

    def peak_tflops(self, dtype: str = "fp16") -> float:
        """Return peak dense tensor-core TFLOPS for the given dtype."""
        if dtype not in self.peak_flops_by_dtype:
            supported = list(self.peak_flops_by_dtype)
            raise ValueError(
                f"dtype '{dtype}' not supported on {self.name} "
                f"(CC {self.compute_capability[0]}.{self.compute_capability[1]}). "
                f"Supported: {supported}"
            )
        return self.peak_flops_by_dtype[dtype] / 1e12


def _warn_unknown(cc: tuple[int, int]) -> None:
    warnings.warn(
        f"Unknown compute capability {cc[0]}.{cc[1]}; MFU/MBU results may be inaccurate. "
        "Please open an issue at https://github.com/Jazhyc/mfu-tracker.",
        UserWarning,
        stacklevel=3,
    )


def _fp8_supported(cc: tuple[int, int]) -> bool:
    major, minor = cc
    return major >= 9 or (major == 8 and minor >= _ADA_FP8_MINOR)


def get_gpu_spec(device: Optional[torch.device] = None) -> GPUSpec:
    """Query the GPU and derive peak throughput from first principles."""
    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    cc = (props.major, props.minor)

    fp16_flops_per_sm = _FP16_FLOPS_PER_SM_PER_CLOCK.get(cc)
    if fp16_flops_per_sm is None:
        # Try falling back to the same major with a known minor
        fallback = next(
            (v for (maj, _), v in _FP16_FLOPS_PER_SM_PER_CLOCK.items() if maj == props.major),
            None,
        )
        _warn_unknown(cc)
        fp16_flops_per_sm = fallback if fallback is not None else 1024

    # clock_rate is reported in kHz
    clock_hz = props.clock_rate * 1_000
    fp16_peak = props.multi_processor_count * fp16_flops_per_sm * clock_hz

    # Build per-dtype peak FLOPS table for this GPU
    peak_flops_by_dtype: dict[str, float] = {}
    for dtype, (min_major, multiplier) in _DTYPE_MULTIPLIER.items():
        if dtype == "fp8":
            if not _fp8_supported(cc):
                continue
        elif props.major < min_major:
            continue
        peak_flops_by_dtype[dtype] = fp16_peak * multiplier

    # Memory bandwidth: memory_clock_rate in kHz, bus width in bits, DDR = ×2
    mem_bw = props.memory_clock_rate * 1_000 * (props.memory_bus_width / 8) * 2

    return GPUSpec(
        name=props.name,
        compute_capability=cc,
        num_sms=props.multi_processor_count,
        clock_rate_hz=clock_hz,
        memory_bandwidth_bytes_per_sec=mem_bw,
        peak_flops_by_dtype=peak_flops_by_dtype,
        peak_memory_bandwidth_tbs=mem_bw / 1e12,
    )
