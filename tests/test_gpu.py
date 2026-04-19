"""Tests for GPU spec derivation (CPU-only, no real GPU needed)."""
import pytest
from unittest.mock import MagicMock, patch

from mfu_tracker.gpu import get_gpu_spec, _FP16_FLOPS_PER_SM_PER_CLOCK


def _mock_props(major=8, minor=0, sms=108, clock_khz=1_410_000, mem_clock_khz=9751, bus_width=5120, name="A100"):
    """clock_khz matches torch's clock_rate unit (kHz). 1410 MHz = 1_410_000 kHz."""
    props = MagicMock()
    props.major = major
    props.minor = minor
    props.multi_processor_count = sms
    props.clock_rate = clock_khz
    props.memory_clock_rate = mem_clock_khz
    props.memory_bus_width = bus_width
    props.name = name
    return props


@patch("torch.cuda.get_device_properties")
def test_a100_peak_tflops(mock_props):
    """A100: 108 SMs × 1024 FP16 FLOPs/SM/clock × 1410 MHz ≈ 156 TFLOPS."""
    mock_props.return_value = _mock_props(major=8, minor=0, sms=108, clock_khz=1_410_000)
    import torch
    spec = get_gpu_spec(torch.device("cuda:0"))
    assert abs(spec.peak_tflops("fp16") - 156.0) < 2.0, spec.peak_tflops("fp16")


@patch("torch.cuda.get_device_properties")
def test_int8_is_2x_fp16(mock_props):
    mock_props.return_value = _mock_props(major=8, minor=0)
    import torch
    spec = get_gpu_spec(torch.device("cuda:0"))
    assert spec.peak_tflops("int8") == pytest.approx(spec.peak_tflops("fp16") * 2)


@patch("torch.cuda.get_device_properties")
def test_ada_supports_fp8(mock_props):
    mock_props.return_value = _mock_props(major=8, minor=9, sms=142, clock_khz=2_520_000, name="L40S")
    import torch
    spec = get_gpu_spec(torch.device("cuda:0"))
    assert "fp8" in spec.peak_flops_by_dtype


@patch("torch.cuda.get_device_properties")
def test_ampere_consumer_no_fp8(mock_props):
    mock_props.return_value = _mock_props(major=8, minor=6, name="RTX 3090")
    import torch
    spec = get_gpu_spec(torch.device("cuda:0"))
    assert "fp8" not in spec.peak_flops_by_dtype


@patch("torch.cuda.get_device_properties")
def test_unknown_cc_warns(mock_props):
    mock_props.return_value = _mock_props(major=99, minor=0)
    import torch
    with pytest.warns(UserWarning, match="Unknown compute capability"):
        spec = get_gpu_spec(torch.device("cuda:0"))
    assert spec.peak_tflops("fp16") > 0


@patch("torch.cuda.get_device_properties")
def test_hopper_higher_than_ampere(mock_props):
    """Hopper should have significantly higher per-SM throughput than Ampere."""
    import torch
    mock_props.return_value = _mock_props(major=9, minor=0, sms=132, clock_khz=1_980_000, name="H100")
    h100 = get_gpu_spec(torch.device("cuda:0"))

    mock_props.return_value = _mock_props(major=8, minor=0, sms=108, clock_khz=1_410_000, name="A100")
    a100 = get_gpu_spec(torch.device("cuda:0"))

    assert h100.peak_tflops("fp16") > a100.peak_tflops("fp16") * 3
