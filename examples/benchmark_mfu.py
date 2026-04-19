#!/usr/bin/env python3
"""
Benchmark MFU and MBU across batch size, attention implementation, and torch.compile.

Model  : GPT-2 (124M params) from HuggingFace Hub
Device : CUDA (skips automatically without a GPU)

Usage:
    .venv/bin/python examples/benchmark_mfu.py
    .venv/bin/python examples/benchmark_mfu.py --steps 50 --seq-len 1024
"""
import argparse
import sys
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mfu_tracker import track
from mfu_tracker.flops import param_bytes, profile_flops

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DTYPE = torch.float16
VOCAB_SIZE = 50257  # GPT-2 vocab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2", help="HF model id or 'tiny' for a local 6-layer config")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30, help="Steps per experiment (after warmup)")
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, attn_impl: str) -> GPT2LMHeadModel:
    if model_id == "tiny":
        cfg = GPT2Config(
            n_layer=6, n_head=8, n_embd=512,
            attn_implementation=attn_impl,
        )
        return GPT2LMHeadModel(cfg).cuda().to(DTYPE)
    return GPT2LMHeadModel.from_pretrained(
        model_id, attn_implementation=attn_impl
    ).cuda().to(DTYPE)


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
) -> tuple[float, float, float] | None:
    """Returns (mean_mfu, mean_mbu, mean_step_ms), or None on OOM."""
    try:
        model = load_model(model_id, attn_impl)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        sample = {
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
            "labels":    torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
        }

        # Profile before compile — FlopCounterMode needs the original graph.
        # with_backward=True uses the standard 3× convention (fwd + 2× bwd).
        flops = profile_flops(model, kwargs=sample, with_backward=True)
        p_bytes = param_bytes(model)

        if use_compile:
            model = torch.compile(model)

        mfu_vals, mbu_vals, ms_vals = [], [], []

        for step in range(warmup + steps):
            batch = {
                "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
                "labels":    torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device="cuda"),
            }

            optimizer.zero_grad()
            with track(flops, p_bytes, dtype="fp16") as result:
                out = model(**batch)
                out.loss.backward()
            optimizer.step()

            if step >= warmup:
                mfu_vals.append(result.mfu)
                mbu_vals.append(result.mbu)
                ms_vals.append(result.elapsed_sec * 1000)

        return (
            sum(mfu_vals) / len(mfu_vals),
            sum(mbu_vals) / len(mbu_vals),
            sum(ms_vals) / len(ms_vals),
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
    """rows: (label, mfu, mbu, ms_per_step)"""
    col_label = max(len(r[0]) for r in rows) + 2
    print()
    header = f"{'Configuration':<{col_label}}  {'MFU':>6}  {'MBU':>6}  {'ms/step':>8}  MFU"
    print(header)
    print("─" * (len(header) + 22))
    baseline_mfu = rows[0][1]
    for label, mfu, mbu, ms in rows:
        delta = f"({(mfu - baseline_mfu) * 100:+.1f}pp)" if mfu != baseline_mfu else "(baseline)"
        print(
            f"{label:<{col_label}}  {mfu:>5.1%}  {mbu:>5.1%}  {ms:>7.1f}ms  "
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
    print(f"Seq len    : {args.seq_len}")
    print(f"Steps      : {args.warmup} warmup + {args.steps} measured")

    experiments = [
        # (label,                       batch, attn,   compile)
        ("batch=1  | eager",             1,    "eager", False),
        ("batch=4  | eager",             4,    "eager", False),
        ("batch=8  | eager",             8,    "eager", False),
        ("batch=8  | sdpa",              8,    "sdpa",  False),
        ("batch=8  | sdpa + compile",    8,    "sdpa",  True),
        ("batch=16 | sdpa",             16,    "sdpa",  False),
        ("batch=16 | sdpa + compile",   16,    "sdpa",  True),
    ]

    rows = []
    for label, bs, attn, compile_ in experiments:
        print(f"\n  Running: {label} ...", end="", flush=True)
        t0 = time.time()
        result = run(args.model, bs, args.seq_len, attn, compile_, args.steps, args.warmup)
        elapsed = time.time() - t0
        if result is None:
            print(f" OOM — skipped")
        else:
            mfu, mbu, ms = result
            rows.append((label, mfu, mbu, ms))
            print(f" done ({elapsed:.0f}s)  MFU={mfu:.1%}  MBU={mbu:.1%}")

    print_table(rows)


if __name__ == "__main__":
    main()
