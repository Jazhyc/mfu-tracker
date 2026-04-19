"""Tests for FLOP counting."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from mfu_tracker.flops import (
    flash_attn_flops,
    param_bytes,
    profile_flops,
    _profile_with_flop_counter,
    _profile_with_thop,
)


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# profile_flops — basic contract
# ---------------------------------------------------------------------------

def test_profile_flops_forward_only():
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    flops = profile_flops(model, args=(sample,), with_backward=False)
    assert flops > 0


def test_profile_flops_with_backward_3x():
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    fwd = profile_flops(model, args=(sample,), with_backward=False)
    total = profile_flops(model, args=(sample,), with_backward=True)
    assert total == fwd * 3


def test_profile_flops_flop_counter_and_thop_agree():
    """FlopCounterMode and thop should return the same count for plain linear ops."""
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    fc_flops = _profile_with_flop_counter(model, (sample,))
    thop_flops = _profile_with_thop(model, (sample,))
    # Should be identical for ops both tools understand
    assert fc_flops == thop_flops


def test_profile_flops_falls_back_to_thop_on_failure():
    """If FlopCounterMode returns -1, thop fallback is used."""
    model = _TinyMLP()
    sample = torch.randn(1, 64)

    with patch("mfu_tracker.flops._profile_with_flop_counter", return_value=-1):
        flops = profile_flops(model, args=(sample,), with_backward=False)

    assert flops == _profile_with_thop(model, (sample,))


# ---------------------------------------------------------------------------
# param_bytes
# ---------------------------------------------------------------------------

def test_param_bytes():
    model = _TinyMLP()
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert param_bytes(model) == expected
    assert param_bytes(model) > 0


def test_param_bytes_trainable_only():
    model = _TinyMLP()
    for p in model.fc1.parameters():
        p.requires_grad_(False)
    total = param_bytes(model)
    trainable = param_bytes(model, trainable_only=True)
    assert trainable < total
    expected = sum(p.numel() * p.element_size() for p in model.fc2.parameters())
    assert trainable == expected


# ---------------------------------------------------------------------------
# flash_attn_flops — analytical formula
# ---------------------------------------------------------------------------

def test_flash_attn_flops_causal_formula():
    B, S, H, D = 2, 512, 8, 64
    flops = flash_attn_flops(B, S, H, D, causal=True)
    assert flops == 2 * B * S * S * H * D


def test_flash_attn_flops_bidirectional_double():
    B, S, H, D = 2, 512, 8, 64
    assert flash_attn_flops(B, S, H, D, causal=False) == flash_attn_flops(B, S, H, D, causal=True) * 2


def test_flash_attn_flops_with_backward_3x():
    B, S, H, D = 1, 256, 4, 32
    fwd = flash_attn_flops(B, S, H, D, with_backward=False)
    assert flash_attn_flops(B, S, H, D, with_backward=True) == fwd * 3


# ---------------------------------------------------------------------------
# SDPA counting on CUDA (skipped without GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDPA dispatch")
def test_flop_counter_counts_sdpa_on_cuda():
    """FlopCounterMode must count SDPA FLOPs on CUDA (thop returns 0 for them)."""
    import torch.nn.functional as F

    B, H, S, D = 2, 4, 128, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    class _SDPAModule(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    model = _SDPAModule().cuda()
    fc_flops = _profile_with_flop_counter(model, (q, k, v))
    thop_flops = _profile_with_thop(model, (q, k, v))

    # FlopCounterMode should count SDPA; thop should miss it
    assert fc_flops > 0, "FlopCounterMode returned 0 for SDPA on CUDA"
    assert fc_flops > thop_flops, (
        f"FlopCounterMode ({fc_flops}) should exceed thop ({thop_flops}) "
        "since thop misses the fused SDPA kernel"
    )
