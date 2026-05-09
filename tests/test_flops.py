"""Tests for FLOP counting."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from mfu_tracker.flops import (
    FlopProfile,
    flash_attn_flops,
    param_bytes,
    profile_flops,
    profile_flops_with_hfu,
    _profile_with_flop_counter,
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


def test_profile_flops_linear_matches_theory():
    """For nn.Linear(K, N) with input (M, K): FLOPs = 2 * M * N * K (+ bias add)."""
    M, K, N = 4, 64, 128
    model = nn.Linear(K, N, bias=False)
    sample = torch.randn(M, K)
    flops, savings = _profile_with_flop_counter(model, (sample,))
    assert flops == 2 * M * N * K
    assert savings == 0  # No SDPA → no causal savings


def test_profile_flops_raises_when_counter_fails():
    """If FlopCounterMode returns -1, profile_flops raises with a helpful message."""
    model = _TinyMLP()
    sample = torch.randn(1, 64)

    with patch("mfu_tracker.flops._profile_with_flop_counter", return_value=(-1, 0)):
        with pytest.raises(RuntimeError, match="FlopCounterMode failed"):
            profile_flops(model, args=(sample,), with_backward=False)


# ---------------------------------------------------------------------------
# profile_flops_with_hfu — HFU equals MFU when no causal SDPA is used
# ---------------------------------------------------------------------------

def test_profile_flops_with_hfu_no_sdpa():
    """For models without SDPA, hfu_flops should equal flops."""
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    profile = profile_flops_with_hfu(model, args=(sample,), with_backward=False)
    assert isinstance(profile, FlopProfile)
    assert profile.flops > 0
    assert profile.hfu_flops == profile.flops


def test_profile_flops_with_hfu_backward_3x():
    """Backward multiplier applies equally to both flop counts."""
    model = _TinyMLP()
    sample = torch.randn(1, 64)
    fwd = profile_flops_with_hfu(model, args=(sample,), with_backward=False)
    total = profile_flops_with_hfu(model, args=(sample,), with_backward=True)
    assert total.flops == fwd.flops * 3
    assert total.hfu_flops == fwd.hfu_flops * 3


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
    """FlopCounterMode must count SDPA FLOPs on CUDA."""
    import torch.nn.functional as F

    B, H, S, D = 2, 4, 128, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    class _SDPAModule(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    model = _SDPAModule().cuda()
    fc_flops, causal_savings = _profile_with_flop_counter(model, (q, k, v))

    # SDPA forward = two matmuls (Q@Kᵀ and attn@V) at 2*B*S²*H*D each.
    # FlopCounterMode reports raw matmul FLOPs without applying the causal mask discount.
    expected = 4 * B * S * S * H * D
    assert fc_flops > 0, "FlopCounterMode returned 0 for SDPA on CUDA"
    assert abs(fc_flops - expected) / expected < 0.05, (
        f"FlopCounterMode ({fc_flops}) deviates >5% from analytical estimate ({expected})"
    )
    # is_causal=True → half the FLOPs are accounted as causal savings.
    assert causal_savings == fc_flops // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDPA dispatch")
def test_hfu_halves_causal_sdpa_only():
    """profile_flops_with_hfu halves causal SDPA but not non-causal SDPA."""
    import torch.nn.functional as F

    B, H, S, D = 2, 4, 128, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    class _CausalSDPA(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    class _NonCausalSDPA(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    causal = profile_flops_with_hfu(_CausalSDPA().cuda(), args=(q, k, v), with_backward=False)
    non_causal = profile_flops_with_hfu(_NonCausalSDPA().cuda(), args=(q, k, v), with_backward=False)

    # Same total FLOPs (PaLM convention) regardless of mask.
    assert causal.flops == non_causal.flops
    # HFU halves causal; non-causal is unchanged.
    assert causal.hfu_flops == causal.flops // 2
    assert non_causal.hfu_flops == non_causal.flops
