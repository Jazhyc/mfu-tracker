"""Tests for MFUCallback — no real GPU needed, CUDA calls mocked."""
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


def test_hfu_backward_factor_autodetected_from_grad_ckpt_off():
    """No gradient checkpointing → hfu_backward_factor=2.0."""
    cb = _make_callback()
    assert cb.hfu_backward_factor is None  # unresolved at init
    args = MagicMock(gradient_checkpointing=False)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.hfu_backward_factor == 2.0


def test_hfu_backward_factor_autodetected_from_grad_ckpt_on():
    """gradient_checkpointing=True → hfu_backward_factor=3.0 (Megatron convention)."""
    cb = _make_callback()
    args = MagicMock(gradient_checkpointing=True)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.hfu_backward_factor == 3.0


def test_hfu_backward_factor_explicit_value_not_overridden():
    """User-supplied float survives auto-detection."""
    sample = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=sample, dtype="fp16", hfu_backward_factor=2.5)
    args = MagicMock(gradient_checkpointing=True)  # would normally force 3.0
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.hfu_backward_factor == 2.5


def test_dtype_autodetected_from_bf16_arg():
    """args.bf16=True → dtype='bf16'."""
    sample = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=sample)  # no dtype passed
    assert cb.dtype is None
    args = MagicMock(bf16=True, fp16=False, gradient_checkpointing=False)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.dtype == "bf16"


def test_dtype_autodetected_from_fp16_arg():
    """args.fp16=True (and bf16 false) → dtype='fp16'."""
    sample = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=sample)
    args = MagicMock(bf16=False, fp16=True, gradient_checkpointing=False)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.dtype == "fp16"


def test_dtype_falls_back_to_fp32_when_no_flag_set():
    """Neither bf16 nor fp16 set → dtype='fp32'."""
    sample = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=sample)
    args = MagicMock(bf16=False, fp16=False, gradient_checkpointing=False)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.dtype == "fp32"


def test_dtype_explicit_value_not_overridden():
    """User-supplied dtype survives auto-detection (e.g. for int8/fp8 inference)."""
    sample = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=sample, dtype="int8")
    args = MagicMock(bf16=True, fp16=False, gradient_checkpointing=False)
    cb.on_train_begin(args, _dummy_state(), _dummy_control(), model=None)
    assert cb.dtype == "int8"


def test_sample_batch_autograbbed_from_dataloader():
    """When sample_batch=None, on_train_begin should pull one from train_dataloader."""
    cb = MFUCallback(dtype="fp16")  # no sample_batch passed
    assert cb.sample_batch is None
    auto_batch = {"x": torch.randn(2, 8)}
    train_dataloader = [auto_batch]  # iterable yielding one batch

    model = _TinyMLP()
    with patch.object(cb, "_profile") as mock_profile:
        cb.on_train_begin(
            MagicMock(gradient_checkpointing=False),
            _dummy_state(),
            _dummy_control(),
            model=model,
            train_dataloader=train_dataloader,
        )

    assert cb.sample_batch is auto_batch
    mock_profile.assert_called_once_with(model)


def test_sample_batch_explicit_not_overridden_by_autograb():
    """User-supplied sample_batch survives even when train_dataloader is available."""
    explicit = {"x": torch.randn(1, 8)}
    cb = MFUCallback(sample_batch=explicit, dtype="fp16")
    train_dataloader = [{"x": torch.randn(99, 8)}]

    with patch.object(cb, "_profile"):
        cb.on_train_begin(
            MagicMock(gradient_checkpointing=False),
            _dummy_state(),
            _dummy_control(),
            model=_TinyMLP(),
            train_dataloader=train_dataloader,
        )

    assert cb.sample_batch is explicit


def test_sample_batch_autograb_warns_on_non_dict():
    """Non-dict batches (e.g. tuples/tensors) skip auto-grab with a warning."""
    cb = MFUCallback(dtype="fp16")
    train_dataloader = [(torch.randn(1, 8), torch.tensor([0]))]  # tuple, not dict

    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cb.on_train_begin(
            MagicMock(gradient_checkpointing=False),
            _dummy_state(),
            _dummy_control(),
            model=_TinyMLP(),
            train_dataloader=train_dataloader,
        )

    assert cb.sample_batch is None
    assert any("non-dict batch" in str(w.message) for w in caught)
    # _profile must NOT be called since sample_batch stayed None
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


def test_pre_optimizer_step_ends_timing_window():
    """Timing should end at on_pre_optimizer_step, not on_step_end."""
    cb = _make_callback()
    cb._fwd_flops = 1_000_000
    cb._spec = MagicMock()
    cb._param_bytes = 50_000

    e_start, e_end = MagicMock(), MagicMock()
    with patch("torch.cuda.Event", side_effect=[e_start, e_end]):
        cb.on_step_begin(_dummy_args(), _dummy_state(), _dummy_control())
        cb.on_pre_optimizer_step(_dummy_args(), _dummy_state(), _dummy_control())
        # on_step_end fires too but should be a no-op since timing already ended
        cb.on_step_end(_dummy_args(), _dummy_state(), _dummy_control())

    e_end.record.assert_called_once()  # only recorded once, by on_pre_optimizer_step
    assert len(cb._pending) == 1


def test_on_step_end_fallback_when_no_pre_optimizer_step():
    """If on_pre_optimizer_step never fires (older HF), on_step_end ends timing."""
    cb = _make_callback()
    cb._fwd_flops = 1_000_000
    cb._spec = MagicMock()
    cb._param_bytes = 50_000

    e_start, e_end = MagicMock(), MagicMock()
    with patch("torch.cuda.Event", side_effect=[e_start, e_end]):
        cb.on_step_begin(_dummy_args(), _dummy_state(), _dummy_control())
        # No on_pre_optimizer_step call — simulate older HF
        cb.on_step_end(_dummy_args(), _dummy_state(), _dummy_control())

    e_end.record.assert_called_once()
    assert len(cb._pending) == 1


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
