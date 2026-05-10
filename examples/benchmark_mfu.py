#!/usr/bin/env python3
"""
Benchmark MFU, HFU, and MBU across batch size, attention implementation,
and torch.compile.

Model  : GPT-2 medium (355M) at bf16 by default — comfortable fit on 16 GB
         across the full batch=1..16 sweep. ~3× the params of gpt2-small,
         which is enough to show meaningfully higher MFU.
Device : CUDA (skips automatically without a GPU)

GPT-2 uses causal SDPA, so HFU < MFU. The gap (MFU - HFU) reflects the work
that a flash-attention kernel skips because of the causal mask. At seq_len=512
the gap is small (attention is a minor fraction of total FLOPs); it grows as
seq_len increases.

Memory headroom on a 16 GB card with vanilla AdamW (fp32 m+v = 8 bytes/param):
    gpt2 (124M)        : ~1 GB before activations  ✓ all batches
    gpt2-medium (355M) : ~4 GB before activations  ✓ all batches (default)
    gpt2-large (774M)  : ~9 GB before activations  ⚠ batch≥8 may swap to CPU
    gpt2-xl (1.5B)     : ~18 GB before activations ✗ won't fit

For larger models on 16 GB, use bitsandbytes.AdamW8bit or activation
checkpointing — beyond the scope of this benchmark.

If the standalone `flash_attn` package (Tri Dao's library) is importable, two
extra rows compare HF's `attn_implementation="flash_attention_2"` head-to-head
with PyTorch's bundled SDPA flash kernel. Otherwise those rows are skipped.

Two final rows demonstrate full activation checkpointing — MFU drops (more
wall time per step due to forward replay) while HFU rises (more hardware FLOPs
counted), giving the canonical Megatron-LM HFU/MFU = 4/3 ratio. Memory drops
dramatically since activations aren't stored, so batch=16 fits where it
previously spilled.

Usage:
    .venv/bin/python examples/benchmark_mfu.py
    .venv/bin/python examples/benchmark_mfu.py --model gpt2 --dtype fp16
    .venv/bin/python examples/benchmark_mfu.py --model gpt2-large  # smaller batches recommended
"""
import argparse
import sys
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mfu_tracker import track
from mfu_tracker.flops import param_bytes, profile_flops_with_hfu

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOCAB_SIZE = 50257  # GPT-2 vocab

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# `attn_implementation="flash_attention_2"` requires the `flash_attn` PyPI
# package (Tri Dao's standalone CUDA library). Detect at import time so the
# experiment is added only when usable.
try:
    import flash_attn  # noqa: F401
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="gpt2-medium",
        help="HF model id (gpt2 / gpt2-medium / gpt2-large / gpt2-xl) or 'tiny' for a local 6-layer config",
    )
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30, help="Steps per experiment (after warmup)")
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, attn_impl: str, dtype: torch.dtype) -> GPT2LMHeadModel:
    # Load directly in the target dtype rather than casting after — avoids the
    # "Flash Attention 2 only supports float16/bfloat16" warning that fires
    # when the model is briefly fp32 between load and `.to(dtype)`.
    if model_id == "tiny":
        cfg = GPT2Config(
            n_layer=6, n_head=8, n_embd=512,
            attn_implementation=attn_impl,
        )
        return GPT2LMHeadModel(cfg).to(dtype).cuda()
    return GPT2LMHeadModel.from_pretrained(
        model_id, attn_implementation=attn_impl, dtype=dtype
    ).cuda()


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run(
    model_id: str,
    batch_size: int,
    seq_len: int,
    attn_impl: str,
    use_compile: bool,
    steps: int,
    warmup: int,
    dtype_str: str,
    grad_ckpt: bool = False,
) -> tuple[float, float, float, float, float] | None:
    """Returns (mean_mfu, mean_hfu, mean_mbu, mean_tokens_per_sec, peak_gb), or None on OOM."""
    dtype = _DTYPE_MAP[dtype_str]
    try:
        model = load_model(model_id, attn_impl, dtype)
        if grad_ckpt:
            # HF: forward replay during backward → +1× forward FLOPs of hardware
            # work per step. MFU is unchanged (algorithmic), HFU gets the 4×
            # multiplier (Megatron convention: HFU/MFU = 4/3 under full ckpt).
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # required when ckpt is on
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sample = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
            "labels":    torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
        }

        # Profile forward only, then apply the correct multiplier per metric:
        # MFU stays at 3× (algorithmic). HFU uses (1 + hfu_backward_factor):
        # 3× without ckpt, 4× with full activation checkpointing.
        profile_fwd = profile_flops_with_hfu(model, kwargs=sample, with_backward=False)
        mfu_flops = profile_fwd.flops * 3
        hfu_backward_factor = 3.0 if grad_ckpt else 2.0
        hfu_flops = int(profile_fwd.hfu_flops * (1 + hfu_backward_factor))
        p_bytes = param_bytes(model)

        if use_compile:
            model = torch.compile(model)

        tokens_per_step = batch_size * seq_len
        mfu_vals, hfu_vals, mbu_vals, tok_s_vals = [], [], [], []

        for step in range(warmup + steps):
            batch = {
                "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
                "labels":    torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
            }

            # Reset the peak counter at the warmup→measured boundary so the
            # reported number reflects steady-state usage during measurement.
            if step == warmup:
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad()
            with track(
                mfu_flops, p_bytes,
                hfu_flop_count=hfu_flops,
                num_tokens=tokens_per_step,
                dtype=dtype_str,
            ) as result:
                out = model(**batch)
                out.loss.backward()
            optimizer.step()

            if step >= warmup:
                mfu_vals.append(result.mfu)
                hfu_vals.append(result.hfu)
                mbu_vals.append(result.mbu)
                tok_s_vals.append(result.tokens_per_sec)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return (
            sum(mfu_vals) / len(mfu_vals),
            sum(hfu_vals) / len(hfu_vals),
            sum(mbu_vals) / len(mbu_vals),
            sum(tok_s_vals) / len(tok_s_vals),
            peak_gb,
        )

    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        try:
            del model, optimizer
        except NameError:
            pass
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_table(rows: list[tuple]) -> None:
    """rows: (label, mfu, hfu, mbu, tok_per_sec, peak_gb)

    Throughput is reported as tokens/sec (= batch × seq_len / step_time) so
    rows with different batch sizes are directly comparable. Raw ms/step
    favors smaller batches and obscures whether more batch buys throughput.
    """
    col_label = max(len(r[0]) for r in rows) + 2
    print()
    header = (
        f"{'Configuration':<{col_label}}  {'MFU':>6}  {'HFU':>6}  {'MBU':>6}  "
        f"{'tok/s':>8}  {'peak GB':>8}  MFU"
    )
    print(header)
    print("─" * (len(header) + 22))
    baseline_mfu = rows[0][1]
    for label, mfu, hfu, mbu, tok_s, peak_gb in rows:
        delta = f"({(mfu - baseline_mfu) * 100:+.1f}pp)" if mfu != baseline_mfu else "(baseline)"
        # Format as e.g. "31.8K"
        tok_s_str = f"{tok_s / 1000:.1f}K"
        print(
            f"{label:<{col_label}}  {mfu:>5.1%}  {hfu:>5.1%}  {mbu:>5.1%}  "
            f"{tok_s_str:>8}  {peak_gb:>7.2f}G  "
            f"{bar(min(mfu, 1.0))} {delta}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU found — skipping benchmark.")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU        : {gpu_name}")
    print(f"Model      : {args.model}")
    print(f"dtype      : {args.dtype}")
    print(f"Seq len    : {args.seq_len}")
    print(f"Steps      : {args.warmup} warmup + {args.steps} measured")

    # batch=16 omitted: triggers CPU spillover on 16 GB cards with gpt2-medium
    # at this seq_len (activations push past VRAM, host fallback dominates).
    experiments = [
        # (label,                       batch, attn,   compile, ckpt)
        ("batch=1  | eager",             1,    "eager", False,  False),
        ("batch=4  | eager",             4,    "eager", False,  False),
        ("batch=8  | eager",             8,    "eager", False,  False),
        ("batch=8  | sdpa",              8,    "sdpa",  False,  False),
        ("batch=8  | sdpa + compile",    8,    "sdpa",  True,   False),
    ]
    # If the standalone flash_attn package is installed, add a head-to-head row
    # against PyTorch's bundled flash kernel (sdpa). Same shapes, same compile
    # status — only the attention kernel differs.
    if _HAS_FLASH_ATTN:
        experiments.append(
            ("batch=8  | flash_attention_2",            8, "flash_attention_2", False, False)
        )
        experiments.append(
            ("batch=8  | flash_attention_2 + compile",  8, "flash_attention_2", True,  False)
        )
    else:
        print("\nflash_attn package not installed — skipping flash_attention_2 rows.")

    # Activation checkpointing: HFU rises to ~4/3 × MFU (Megatron convention).
    # The "ckpt only" row (no compile) is the textbook case — expect step time
    # to grow ~33% vs the matching no-ckpt sdpa row, dragging MFU down. The
    # "compile + ckpt" rows test the hypothesis that torch.compile already
    # rematerializes aggressively, making explicit checkpointing nearly free.
    # batch=16 ckpt fits where it would otherwise spill — activations aren't
    # stored, so peak memory drops.
    experiments.append(
        ("batch=8  | sdpa + ckpt",            8, "sdpa", False, True)
    )
    experiments.append(
        ("batch=8  | sdpa + compile + ckpt",  8, "sdpa", True,  True)
    )
    experiments.append(
        ("batch=16 | sdpa + compile + ckpt", 16, "sdpa", True,  True)
    )
    # Best-of-everything row: largest feasible batch with flash_attention_2,
    # torch.compile, and gradient checkpointing all stacked. If flash_attn isn't
    # installed, skip — the equivalent sdpa row above is the next best thing.
    if _HAS_FLASH_ATTN:
        experiments.append(
            ("batch=16 | flash_attention_2 + compile + ckpt", 16, "flash_attention_2", True, True)
        )

    rows = []
    for label, bs, attn, compile_, ckpt in experiments:
        print(f"\n  Running: {label} ...", end="", flush=True)
        t0 = time.time()
        result = run(args.model, bs, args.seq_len, attn, compile_, args.steps, args.warmup, args.dtype, grad_ckpt=ckpt)
        elapsed = time.time() - t0
        if result is None:
            print(" OOM — skipped")
        else:
            mfu, hfu, mbu, tok_s, peak_gb = result
            rows.append((label, mfu, hfu, mbu, tok_s, peak_gb))
            print(
                f" done ({elapsed:.0f}s)  MFU={mfu:.1%}  HFU={hfu:.1%}  "
                f"MBU={mbu:.1%}  {tok_s/1000:.1f}K tok/s  peak={peak_gb:.2f}G"
            )

    print_table(rows)


if __name__ == "__main__":
    main()
