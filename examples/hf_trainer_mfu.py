#!/usr/bin/env python3
"""
HuggingFace Trainer + MFUCallback example.

Trains GPT-2 (124M) on synthetic data to demonstrate that MFU and MBU appear
automatically in Trainer logs (and WandB if enabled).

Expected MFU on a modern GPU (RTX 4080): ~5–12% depending on batch size.
MFU will be near-zero on CPU — a GPU is required for meaningful numbers.

Usage:
    .venv/bin/python examples/hf_trainer_mfu.py
    .venv/bin/python examples/hf_trainer_mfu.py --dtype bf16 --batch-size 16
    WANDB_PROJECT=mfu-test .venv/bin/python examples/hf_trainer_mfu.py --wandb

Metrics appear under "throughput/mfu" and "throughput/mbu" in WandB, grouped
in their own section away from loss/lr. Override with --metric-prefix.
"""
import argparse
import sys

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

from mfu_tracker.integrations.hf_trainer import MFUCallback

VOCAB_SIZE = 50257  # standard GPT-2 vocab
SEQ_LEN = 512


class _SyntheticDataset(Dataset):
    """Random token sequences — no download needed."""

    def __init__(self, size: int = 512):
        self._size = size
        self._data = torch.randint(0, VOCAB_SIZE, (size, SEQ_LEN))

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        ids = self._data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--steps", type=int, default=60, help="Max training steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    p.add_argument(
        "--metric-prefix",
        default="throughput",
        help='WandB metric prefix (default "throughput" → throughput/mfu)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU found — MFU/MBU will not be logged (requires a GPU).")
        sys.exit(0)

    # GPT-2 small (124M) — same architecture as the benchmark script.
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    model = GPT2LMHeadModel(config)

    dataset = _SyntheticDataset(size=512)
    # Sample batch stays on CPU; MFUCallback moves it to the model device before profiling.
    sample_batch = {k: v[:args.batch_size] for k, v in dataset[:args.batch_size].items()}

    # MFUCallback profiles forward FLOPs once at on_train_begin, then records
    # CUDA events each step and logs throughput/mfu + throughput/mbu at every
    # logging interval. WandB (if active) receives these alongside loss/lr with
    # no extra configuration.
    callback = MFUCallback(
        sample_batch=sample_batch,
        dtype=args.dtype,
        metric_prefix=args.metric_prefix,
    )

    fp16 = args.dtype == "fp16"
    bf16 = args.dtype == "bf16"

    training_args = TrainingArguments(
        output_dir="/tmp/mfu-trainer-example",
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        report_to=["wandb"] if args.wandb else ["none"],
        fp16=fp16,
        bf16=bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[callback],
    )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU   : {gpu_name}")
    print(f"Model : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params  (GPT-2 small)")
    print(f"dtype : {args.dtype}  |  batch={args.batch_size}  seq={SEQ_LEN}")
    print(f"steps : {args.steps}  (logging every {args.logging_steps})")
    if args.wandb:
        print("WandB : enabled — metrics appear under 'throughput/' section")
    else:
        print("WandB : disabled (pass --wandb to enable)")
    print()

    trainer.train()

    # MFU and MBU appear inline in the training progress above.
    # With --wandb they are additionally sent to the "throughput/" WandB section.


if __name__ == "__main__":
    main()
