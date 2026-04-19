"""Tests for MFUCallback — no real GPU needed, CUDA calls mocked."""
import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from mfu_tracker.integrations.hf_trainer import MFUCallback


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


def _make_callback(model=None):
    sample = {"x": torch.randn(1, 8)}
    return MFUCallback(sample_batch=sample, dtype="fp16")


def _dummy_args():
    return MagicMock()


def _dummy_state():
    return MagicMock()


def _dummy_control():
    return MagicMock()


def test_on_train_begin_profiles_model():
    cb = _make_callback()
    model = _TinyMLP()

    with patch.object(cb, "_profile") as mock_profile:
        cb.on_train_begin(_dummy_args(), _dummy_state(), _dummy_control(), model=model)

    mock_profile.assert_called_once_with(model)
    assert cb._model is model


def test_on_train_begin_no_model_is_noop():
    cb = _make_callback()
    cb.on_train_begin(_dummy_args(), _dummy_state(), _dummy_control(), model=None)
    assert cb._model is None
    assert cb._fwd_flops is None


def test_on_step_begin_skipped_when_not_profiled():
    """If profiling failed, on_step_begin should do nothing rather than crash."""
    cb = _make_callback()
    # _fwd_flops is None — not yet profiled
    cb.on_step_begin(_dummy_args(), _dummy_state(), _dummy_control())
    assert not hasattr(cb, "_pending_step")


def test_on_step_begin_registers_hook_and_records_event():
    cb = _make_callback()
    model = _TinyMLP()
    cb._model = model
    cb._fwd_flops = 1_000_000
    cb._spec = MagicMock()
    cb._param_bytes = 50_000

    mock_event = MagicMock()
    with patch("torch.cuda.Event", return_value=mock_event):
        cb.on_step_begin(_dummy_args(), _dummy_state(), _dummy_control())

    assert hasattr(cb, "_pending_step")
    mock_event.record.assert_called()  # e_start.record() was called


def test_step_begin_end_accumulates_pending():
    cb = _make_callback()
    model = _TinyMLP()
    cb._model = model
    cb._fwd_flops = 1_000_000
    cb._spec = MagicMock()
    cb._param_bytes = 50_000

    mock_event = MagicMock()
    with patch("torch.cuda.Event", return_value=mock_event):
        cb.on_step_begin(_dummy_args(), _dummy_state(), _dummy_control())
        cb.on_step_end(_dummy_args(), _dummy_state(), _dummy_control())

    assert len(cb._pending) == 1
    assert not hasattr(cb, "_pending_step")


def test_on_log_computes_mfu_mbu():
    cb = _make_callback()
    cb._fwd_flops = 1_000_000
    cb._param_bytes = 50_000
    cb._spec = MagicMock()
    cb._spec.peak_tflops.return_value = 156.0
    cb._spec.peak_memory_bandwidth_tbs = 2.0

    # Simulate two accumulated steps: 70ms each
    e_start = MagicMock()
    e_end = MagicMock()
    e_start.elapsed_time.return_value = 70.0
    cb._pending = [
        (e_start, e_end),
        (e_start, e_end),
    ]

    logs = {}
    with patch("torch.cuda.synchronize"):
        cb.on_log(_dummy_args(), _dummy_state(), _dummy_control(), logs=logs)

    assert "throughput/mfu" in logs
    assert "throughput/mbu" in logs
    assert isinstance(logs["throughput/mfu"], float)
    assert isinstance(logs["throughput/mbu"], float)
    assert cb._pending == []  # flushed


def test_on_log_skips_when_no_pending():
    cb = _make_callback()
    cb._fwd_flops = 1_000_000
    cb._spec = MagicMock()
    logs = {}
    cb.on_log(_dummy_args(), _dummy_state(), _dummy_control(), logs=logs)
    assert "mfu" not in logs


def test_on_log_skips_when_logs_is_none():
    cb = _make_callback()
    cb._fwd_flops = 1_000_000
    # Should not raise
    cb.on_log(_dummy_args(), _dummy_state(), _dummy_control(), logs=None)
