"""
Integration tests that require a real CUDA GPU.

Run with: .venv/bin/pytest tests/test_integration_gpu.py -v
Skipped automatically in environments without CUDA.
"""
import pytest
import torch
import torch.nn as nn

from mfu_tracker.flops import param_bytes, profile_flops
from mfu_tracker.gpu import get_gpu_spec
from mfu_tracker.tracker import compute_mfu, track

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)


# ---------------------------------------------------------------------------
# GPU spec detection
# ---------------------------------------------------------------------------

def test_gpu_detected_without_warning():
    """Known GPUs should not emit an unknown-CC warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = get_gpu_spec()
    assert spec.peak_tflops("fp16") > 0


def test_peak_tflops_positive_and_finite():
    spec = get_gpu_spec()
    peak = spec.peak_tflops("fp16")
    assert 0 < peak < 10_000, f"Unreasonable peak TFLOPS: {peak}"


def test_memory_bandwidth_positive():
    spec = get_gpu_spec()
    assert spec.peak_memory_bandwidth_tbs > 0


def test_int8_peak_is_2x_fp16():
    spec = get_gpu_spec()
    assert spec.peak_tflops("int8") == pytest.approx(spec.peak_tflops("fp16") * 2)


def test_ada_fp8_supported():
    """RTX 4080 is Ada Lovelace (CC 8.9) — FP8 must be in the spec table."""
    spec = get_gpu_spec()
    major, minor = spec.compute_capability
    if (major, minor) == (8, 9):
        assert "fp8" in spec.peak_flops_by_dtype, (
            "FP8 should be supported on Ada Lovelace (CC 8.9)"
        )


# ---------------------------------------------------------------------------
# FLOP counting accuracy (thop against analytical ground truth)
# ---------------------------------------------------------------------------

def test_linear_flop_count_matches_theory():
    """
    For nn.Linear(K, N, bias=False) with input (M, K):
    FLOPs = 2 * M * N * K  (K MACs per output element, 2 FLOPs per MAC).
    """
    M, K, N = 512, 1024, 2048
    model = nn.Linear(K, N, bias=False)
    x = torch.randn(M, K)

    flops = profile_flops(model, args=(x,), with_backward=False)
    expected = 2 * M * N * K

    rel_error = abs(flops - expected) / expected
    assert rel_error < 0.01, (
        f"thop FLOP count {flops:,} differs from theory {expected:,} "
        f"by {rel_error:.1%}"
    )


def test_conv2d_flop_count_matches_theory():
    """
    For Conv2d(Cin, Cout, K) with input (N, Cin, H, W):
    FLOPs ≈ 2 * N * Cout * H_out * W_out * Cin * K * K.
    """
    Cin, Cout, K = 64, 128, 3
    N, H, W = 4, 32, 32
    H_out = W_out = H - K + 1  # no padding, stride 1

    model = nn.Conv2d(Cin, Cout, K, bias=False)
    x = torch.randn(N, Cin, H, W)

    flops = profile_flops(model, args=(x,), with_backward=False)
    expected = 2 * N * Cout * H_out * W_out * Cin * K * K

    rel_error = abs(flops - expected) / expected
    assert rel_error < 0.01, (
        f"thop FLOP count {flops:,} differs from theory {expected:,} "
        f"by {rel_error:.1%}"
    )


# ---------------------------------------------------------------------------
# End-to-end: real compute step on GPU
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_model_and_spec():
    """Shared fixture: a large-ish fp16 linear on CUDA + pre-queried spec."""
    spec = get_gpu_spec()
    model = nn.Linear(4096, 4096, bias=False).cuda().to(torch.float16)
    x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)

    # Warmup — ensures GPU is at boost clock before we time anything.
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()

    flops = profile_flops(model, args=(x,), with_backward=False)
    p_bytes = param_bytes(model)
    return model, x, flops, p_bytes, spec


def test_mfu_in_unit_interval(gpu_model_and_spec):
    model, x, flops, p_bytes, spec = gpu_model_and_spec

    with track(flops, p_bytes, dtype="fp16", spec=spec) as result:
        for _ in range(20):
            _ = model(x)

    assert result.mfu is not None
    assert 0 < result.mfu <= 1.5, (
        f"MFU {result.mfu:.3f} out of expected range — check peak ceiling formula"
    )


def test_mbu_in_unit_interval(gpu_model_and_spec):
    model, x, flops, p_bytes, spec = gpu_model_and_spec

    with track(flops, p_bytes, dtype="fp16", spec=spec) as result:
        for _ in range(20):
            _ = model(x)

    assert result.mbu is not None
    assert 0 < result.mbu <= 1.5


def test_compute_mfu_matches_track(gpu_model_and_spec):
    """compute_mfu() must agree with track() for the same workload."""
    model, x, flops, p_bytes, spec = gpu_model_and_spec
    n = 20

    with track(flops * n, p_bytes * n, dtype="fp16", spec=spec) as result:
        for _ in range(n):
            _ = model(x)

    assert result.elapsed_sec is not None
    standalone = compute_mfu(flops * n, result.elapsed_sec, dtype="fp16", spec=spec)
    assert result.mfu == pytest.approx(standalone, rel=1e-4)


def test_larger_batch_higher_mfu(gpu_model_and_spec):
    """
    Increasing the batch size moves the op toward compute-bound,
    so MFU should increase (or at minimum not decrease dramatically).
    """
    _, _, _, _, spec = gpu_model_and_spec
    results = {}

    for batch in [4, 64]:
        model = nn.Linear(4096, 4096, bias=False).cuda().to(torch.float16)
        x = torch.randn(batch, 4096, device="cuda", dtype=torch.float16)
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

        flops = profile_flops(model, args=(x,), with_backward=False)
        p_bytes = param_bytes(model)

        with track(flops, p_bytes, dtype="fp16", spec=spec) as result:
            for _ in range(20):
                _ = model(x)

        results[batch] = result.mfu

    assert results[64] > results[4] * 0.8, (
        f"Expected larger batch to have higher or similar MFU: "
        f"batch=4 → {results[4]:.3f}, batch=64 → {results[64]:.3f}"
    )
