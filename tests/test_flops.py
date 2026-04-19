"""Tests for FLOP counting via calflops."""
import torch
import torch.nn as nn

from mfu_tracker.flops import param_bytes, profile_flops


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_profile_flops_forward_only():
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    flops = profile_flops(model, args=(sample,), with_backward=False)
    assert flops > 0


def test_profile_flops_with_backward_is_larger():
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    fwd = profile_flops(model, args=(sample,), with_backward=False)
    total = profile_flops(model, args=(sample,), with_backward=True)
    # backward adds 2× forward, so total ≈ 3× forward
    assert total > fwd


def test_param_bytes():
    model = _TinyMLP()
    # fc1: 64*128 + 128 = 8320 params, fc2: 128*64 + 64 = 8256 params → 16576 total
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert param_bytes(model) == expected
    assert param_bytes(model) > 0
