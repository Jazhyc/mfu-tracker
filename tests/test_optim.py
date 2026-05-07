"""Tests for MFUOptimizerWrapper — CPU-only, CUDA calls mocked."""
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from mfu_tracker.optim import MFUOptimizerWrapper
from mfu_tracker.tracker import UtilizationResult


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_wrapper(model=None):
    model = model or _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sample = {"x": torch.randn(1, 8)}
    return MFUOptimizerWrapper(optimizer, model, sample_batch=sample, dtype="fp16")


def test_proxies_step_and_zero_grad():
    wrapper = _make_wrapper()
    assert hasattr(wrapper, "step")
    assert hasattr(wrapper, "zero_grad")


def test_getattr_proxies_to_optimizer():
    wrapper = _make_wrapper()
    # param_groups is an attribute of the underlying optimizer
    assert wrapper.param_groups is wrapper.optimizer.param_groups


def test_track_step_yields_utilization_result():
    """track_step() must yield a UtilizationResult even if CUDA is unavailable."""
    wrapper = _make_wrapper()

    mock_event = MagicMock()
    mock_event.elapsed_time.return_value = 100.0  # 100 ms total

    with (
        patch("torch.cuda.Event", return_value=mock_event),
        patch("torch.cuda.synchronize"),
        patch.object(wrapper, "_profile_once"),
    ):
        # Simulate a profiled state
        wrapper._spec = MagicMock()
        wrapper._spec.peak_tflops.return_value = 156.0
        wrapper._spec.peak_memory_bandwidth_tbs = 2.0
        wrapper._fwd_flops = 1_000_000
        wrapper._param_bytes = 100_000
        # bwd hook fires: simulate e_bwd recorded at 30ms into step
        mock_event.elapsed_time.return_value = 100.0  # e_start → e_end

        with wrapper.track_step() as result:
            pass  # no real forward/backward needed

    assert isinstance(result, UtilizationResult)


def test_track_step_result_fields_populated():
    wrapper = _make_wrapper()
    mock_event = MagicMock()

    with (
        patch("torch.cuda.Event", return_value=mock_event),
        patch("torch.cuda.synchronize"),
    ):
        wrapper._spec = MagicMock()
        wrapper._spec.peak_tflops.return_value = 156.0
        wrapper._spec.peak_memory_bandwidth_tbs = 2.0
        wrapper._fwd_flops = 1_000_000
        wrapper._param_bytes = 100_000
        mock_event.elapsed_time.return_value = 100.0

        with wrapper.track_step() as result:
            pass

    # Result fields should be populated after access (lazy resolution via CUDA events).
    assert result.elapsed_sec is not None
    assert result.mfu is not None
    assert result.mbu is not None
