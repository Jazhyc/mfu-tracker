"""Numerical correctness tests for track() and compute_mfu/mbu."""
import pytest
from unittest.mock import MagicMock, patch
import torch

from mfu_tracker.tracker import UtilizationResult, compute_mbu, compute_mfu, track


def _mock_spec(peak_tflops=156.0, peak_tbs=2.0):
    spec = MagicMock()
    spec.peak_tflops.return_value = peak_tflops
    spec.peak_memory_bandwidth_tbs = peak_tbs
    return spec


# ---------------------------------------------------------------------------
# compute_mfu / compute_mbu — pure math, no mocking needed
# ---------------------------------------------------------------------------

def test_compute_mfu_exact():
    spec = _mock_spec(peak_tflops=100.0)
    # 50 TFLOPS achieved out of 100 TFLOPS peak → MFU = 0.5
    flops = int(50e12)
    elapsed = 1.0
    assert compute_mfu(flops, elapsed, spec=spec) == pytest.approx(0.5)


def test_compute_mfu_scales_with_time():
    spec = _mock_spec(peak_tflops=100.0)
    flops = int(50e12)
    # Doubling elapsed halves MFU
    mfu_1s = compute_mfu(flops, 1.0, spec=spec)
    mfu_2s = compute_mfu(flops, 2.0, spec=spec)
    assert mfu_1s == pytest.approx(mfu_2s * 2)


def test_compute_mbu_exact():
    spec = _mock_spec(peak_tbs=2.0)
    # 1 TB/s achieved out of 2 TB/s peak → MBU = 0.5
    param_bytes = int(1e12)
    elapsed = 1.0
    assert compute_mbu(param_bytes, elapsed, spec=spec) == pytest.approx(0.5)


def test_compute_mfu_num_gpus_halves_result():
    spec = _mock_spec(peak_tflops=100.0)
    flops = int(50e12)
    mfu_1 = compute_mfu(flops, 1.0, spec=spec, num_gpus=1)
    mfu_2 = compute_mfu(flops, 1.0, spec=spec, num_gpus=2)
    assert mfu_1 == pytest.approx(mfu_2 * 2)


def test_compute_mbu_num_gpus_halves_result():
    spec = _mock_spec(peak_tbs=2.0)
    mbu_1 = compute_mbu(int(1e12), 1.0, spec=spec, num_gpus=1)
    mbu_2 = compute_mbu(int(1e12), 1.0, spec=spec, num_gpus=2)
    assert mbu_1 == pytest.approx(mbu_2 * 2)


def test_track_num_gpus_scales_peak():
    spec = _mock_spec(peak_tflops=100.0, peak_tbs=2.0)
    flops = int(50e12)
    p_bytes = int(1e12)
    elapsed_s = 1.0

    with (
        patch("torch.cuda.synchronize"),
        patch("mfu_tracker.tracker.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = [0.0, elapsed_s]
        with track(flops, p_bytes, spec=spec, num_gpus=2) as result:
            pass

    # Peak doubled → MFU and MBU halved vs single-GPU
    assert result.mfu == pytest.approx(0.25)
    assert result.mbu == pytest.approx(0.25)


def test_num_gpus_default_is_1():
    """Default num_gpus=1 is correct for DDP — per-GPU MFU equals global MFU."""
    spec = _mock_spec(peak_tflops=100.0, peak_tbs=2.0)
    flops = int(50e12)
    p_bytes = int(1e12)

    with (
        patch("torch.cuda.synchronize"),
        patch("mfu_tracker.tracker.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = [0.0, 1.0]
        with track(flops, p_bytes, spec=spec) as result:
            pass

    assert result.num_gpus == 1
    assert result.mfu == pytest.approx(0.5)


def test_mfu_above_one_possible_on_bad_spec():
    """MFU > 1 signals our peak ceiling is wrong — not clamped, so the user sees it."""
    spec = _mock_spec(peak_tflops=10.0)  # underestimated peak
    flops = int(50e12)
    mfu = compute_mfu(flops, 1.0, spec=spec)
    assert mfu > 1.0


# ---------------------------------------------------------------------------
# track() context manager — end-to-end numerical check
# ---------------------------------------------------------------------------

def test_track_mfu_correct():
    spec = _mock_spec(peak_tflops=100.0, peak_tbs=2.0)
    flops = int(50e12)     # 50 TFLOPS
    p_bytes = int(1e12)    # 1 TB
    elapsed_s = 1.0        # 1 second → 50 TFLOPS achieved → MFU 0.5, MBU 0.5

    with (
        patch("torch.cuda.synchronize"),
        patch("mfu_tracker.tracker.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = [0.0, elapsed_s]
        with track(flops, p_bytes, spec=spec) as result:
            pass  # no real work; timing is mocked

    assert result.mfu == pytest.approx(0.5)
    assert result.mbu == pytest.approx(0.5)
    assert result.elapsed_sec == pytest.approx(elapsed_s)
    assert result.achieved_tflops == pytest.approx(50.0)
    assert result.achieved_tbs == pytest.approx(1.0)


def test_track_result_none_during_block():
    """Fields must be None while the block is executing."""
    spec = _mock_spec()
    captured = {}

    with (
        patch("torch.cuda.synchronize"),
        patch("mfu_tracker.tracker.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = [0.0, 1.0]
        with track(int(1e12), int(1e9), spec=spec) as result:
            captured["mfu"] = result.mfu
            captured["mbu"] = result.mbu

    assert captured["mfu"] is None
    assert captured["mbu"] is None
    assert result.mfu is not None  # populated after block


def test_track_and_compute_mfu_agree():
    """track() and compute_mfu() must return the same value for identical inputs."""
    spec = _mock_spec(peak_tflops=200.0, peak_tbs=3.0)
    flops = int(80e12)
    p_bytes = int(1.5e12)
    elapsed_s = 1.0

    with (
        patch("torch.cuda.synchronize"),
        patch("mfu_tracker.tracker.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = [0.0, elapsed_s]
        with track(flops, p_bytes, spec=spec) as result:
            pass

    assert result.mfu == pytest.approx(compute_mfu(flops, elapsed_s, spec=spec))
    assert result.mbu == pytest.approx(compute_mbu(p_bytes, elapsed_s, spec=spec))


# ---------------------------------------------------------------------------
# UtilizationResult lazy resolution (track_step path)
# ---------------------------------------------------------------------------

def test_lazy_result_resolves_on_access():
    spec = _mock_spec(peak_tflops=100.0, peak_tbs=2.0)

    e_start = MagicMock()
    e_bwd = MagicMock()
    e_end = MagicMock()
    # Total 1000ms, forward 400ms → backward 600ms → factor 1.5
    e_start.elapsed_time.side_effect = lambda other: (
        1000.0 if other is e_end else 400.0
    )

    fwd_flops = int(40e12)   # 40 TFLOPS forward
    # total flops = 40e12 * (1 + 1.5) = 100e12 → 100 TFLOPS / 1s → MFU 1.0
    p_bytes = int(1e12)

    result = UtilizationResult(dtype="fp16", gpu_spec=spec)
    result._e_start = e_start
    result._e_bwd = e_bwd
    result._e_end = e_end
    result._bwd_recorded = True
    result._fwd_flops = fwd_flops
    result._param_bytes = p_bytes
    result._device = None

    with patch("torch.cuda.synchronize"):
        assert result.mfu == pytest.approx(1.0)
        assert result.mbu == pytest.approx(0.5)


def test_lazy_result_resolves_only_once():
    """_resolve() must not call synchronize more than once."""
    spec = _mock_spec()
    e_start = MagicMock()
    e_start.elapsed_time.return_value = 1000.0

    result = UtilizationResult(dtype="fp16", gpu_spec=spec)
    result._e_start = e_start
    result._e_bwd = MagicMock()
    result._e_end = MagicMock()
    result._bwd_recorded = False
    result._fwd_flops = int(1e12)
    result._param_bytes = int(1e9)
    result._device = None

    with patch("torch.cuda.synchronize") as mock_sync:
        _ = result.mfu
        _ = result.mfu  # second access
        _ = result.mbu

    mock_sync.assert_called_once()
