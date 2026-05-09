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
    if model_id == "tiny":
        cfg = GPT2Config(
            n_layer=6, n_head=8, n_embd=512,
            attn_implementation=attn_impl,
        )
        return GPT2LMHeadModel(cfg).cuda().to(dtype)
    return GPT2LMHeadModel.from_pretrained(
        model_id, attn_implementation=attn_impl
    ).cuda().to(dtype)


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
) -> tuple[float, float, float, float, float] | None:
    """Returns (mean_mfu, mean_hfu, mean_mbu, mean_step_ms, peak_gb), or None on OOM."""
    dtype = _DTYPE_MAP[dtype_str]
    try:
        model = load_model(model_id, attn_impl, dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sample = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
            "labels":    torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
        }

        # Profile before compile — FlopCounterMode needs the original graph.
        # with_backward=True uses the standard 3× convention (fwd + 2× bwd).
        # profile_flops_with_hfu returns both PaLM-style and causal-corrected
        # counts, and counts flash_attn_func calls automatically when the
        # `flash_attn` package is installed (its ops are in the dispatcher).
        profile = profile_flops_with_hfu(model, kwargs=sample, with_backward=True)
        p_bytes = param_bytes(model)

        if use_compile:
            model = torch.compile(model)

        mfu_vals, hfu_vals, mbu_vals, ms_vals = [], [], [], []

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
            with track(profile.flops, p_bytes, hfu_flop_count=profile.hfu_flops, dtype=dtype_str) as result:
                out = model(**batch)
                out.loss.backward()
            optimizer.step()

            if step >= warmup:
                mfu_vals.append(result.mfu)
                hfu_vals.append(result.hfu)
                mbu_vals.append(result.mbu)
                ms_vals.append(result.elapsed_sec * 1000)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return (
            sum(mfu_vals) / len(mfu_vals),
            sum(hfu_vals) / len(hfu_vals),
            sum(mbu_vals) / len(mbu_vals),
            sum(ms_vals) / len(ms_vals),
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
    """rows: (label, mfu, hfu, mbu, ms_per_step, peak_gb)"""
    col_label = max(len(r[0]) for r in rows) + 2
    print()
    header = (
        f"{'Configuration':<{col_label}}  {'MFU':>6}  {'HFU':>6}  {'MBU':>6}  "
        f"{'ms/step':>8}  {'peak GB':>8}  MFU"
    )
    print(header)
    print("─" * (len(header) + 22))
    baseline_mfu = rows[0][1]
    for label, mfu, hfu, mbu, ms, peak_gb in rows:
        delta = f"({(mfu - baseline_mfu) * 100:+.1f}pp)" if mfu != baseline_mfu else "(baseline)"
        print(
            f"{label:<{col_label}}  {mfu:>5.1%}  {hfu:>5.1%}  {mbu:>5.1%}  "
            f"{ms:>7.1f}ms  {peak_gb:>7.2f}G  "
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
        # (label,                       batch, attn,   compile)
        ("batch=1  | eager",             1,    "eager", False),
        ("batch=4  | eager",             4,    "eager", False),
        ("batch=8  | eager",             8,    "eager", False),
        ("batch=8  | sdpa",              8,    "sdpa",  False),
        ("batch=8  | sdpa + compile",    8,    "sdpa",  True),
    ]
    # If the standalone flash_attn package is installed, add a head-to-head row
    # against PyTorch's bundled flash kernel (sdpa). Same shapes, same compile
    # status — only the attention kernel differs.
    if _HAS_FLASH_ATTN:
        experiments.append(
            ("batch=8  | flash_attention_2", 8, "flash_attention_2", False)
        )
        experiments.append(
            ("batch=8  | flash_attention_2 + compile", 8, "flash_attention_2", True)
        )
    else:
        print("\nflash_attn package not installed — skipping flash_attention_2 rows.")

    rows = []
    for label, bs, attn, compile_ in experiments:
        print(f"\n  Running: {label} ...", end="", flush=True)
        t0 = time.time()
        result = run(args.model, bs, args.seq_len, attn, compile_, args.steps, args.warmup, args.dtype)
        elapsed = time.time() - t0
        if result is None:
            print(" OOM — skipped")
        else:
            mfu, hfu, mbu, ms, peak_gb = result
            rows.append((label, mfu, hfu, mbu, ms, peak_gb))
            print(
                f" done ({elapsed:.0f}s)  MFU={mfu:.1%}  HFU={hfu:.1%}  "
                f"MBU={mbu:.1%}  peak={peak_gb:.2f}G"
            )

    print_table(rows)


if __name__ == "__main__":
    main()
