# mfu-tracker

When profiling training runs, I found that most existing tools either lacked MFU/MBU support entirely or dragged in hundreds of megabytes of transitive dependencies. This library is an attempt at a self-contained alternative.

**mfu-tracker** is a PyTorch library for measuring Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU). It supports bare PyTorch training loops, an optimizer wrapper, and a HuggingFace Trainer callback.

- **Minimal dependencies** — PyTorch only.
- **Profiled FLOPs, not formula estimates** — uses `FlopCounterMode` to count the FLOPs your model executes rather than an approximation like `6 × params × tokens`. For Mixture-of-Experts models this means only active experts are counted, giving a more accurate numerator than parameter-based estimates.
- **Three integration styles** — context manager, optimizer wrapper, HF Trainer callback.
- **WandB / TensorBoard / MLflow** — metrics are logged through HF Trainer's existing pipeline when using `MFUCallback`.

MFU as a training efficiency metric was introduced in the [PaLM paper](https://arxiv.org/abs/2204.02311) (Chowdhery et al., 2022).

---

## What MFU and MBU measure

**MFU (Model FLOPs Utilization)** is the ratio of observed FLOP throughput to the GPU's theoretical peak for the given dtype. A value of 0.50 means the model is executing at half the GPU's rated peak. Well-optimized large models on modern hardware typically fall in the 0.40–0.60 range; small models often land much lower due to kernel dispatch overhead relative to compute time.

**MBU (Model Bandwidth Utilization)** as computed here is a proxy, not a direct DRAM measurement. It is defined as:

```
MBU = (param_bytes / elapsed_sec) / peak_memory_bandwidth
```

where `param_bytes` is the total size of model parameters and `elapsed_sec` is wall time. This assumes one full pass through model weights per step and does not account for activation memory, gradients, optimizer state, or data layout effects. It is most useful as a relative indicator across runs rather than an absolute efficiency measure.

If both MFU and MBU are low simultaneously, the GPU is underutilized. Two common causes: kernel dispatch overhead (the CPU cannot issue kernels fast enough to keep the GPU busy — `torch.compile` reduces this by fusing operations), or CPU-side pipeline stalls (such as a slow DataLoader).

---

## Installation

```bash
pip install mfu-tracker
```

HuggingFace Trainer integration requires no extra install — if you are already running HF Trainer, `transformers` is already available. Import `MFUCallback` directly.

---

## Usage

### HuggingFace Trainer

```python
from mfu_tracker.integrations.hf_trainer import MFUCallback

# Zero-config: sample_batch is grabbed from train_dataloader at on_train_begin;
# dtype is read from TrainingArguments.bf16 / args.fp16; hfu_backward_factor
# is read from args.gradient_checkpointing.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[MFUCallback()],
)
trainer.train()
```

Pass `sample_batch=` explicitly only if you want to control the calibration shape — for example, if your real training batches are very large and you'd rather profile on a single example, or if your dataset is an `IterableDataset` and you don't want one batch consumed for calibration.

`throughput/mfu`, `throughput/mbu`, and `throughput/tokens_per_sec` (when the batch has an `input_ids` field) are added to the Trainer log dict at each logging step and forwarded automatically to any configured integrations (WandB, TensorBoard, MLflow). WandB groups metrics by the `/` separator, so these appear in a distinct "throughput" section rather than alongside loss and learning rate. `throughput/hfu` is also emitted when it would differ from MFU (causal SDPA or activation checkpointing).

### Optimizer wrapper

```python
from mfu_tracker import MFUOptimizerWrapper

base_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer = MFUOptimizerWrapper(
    base_optimizer, model,
    sample_batch={"input_ids": sample_ids},
    dtype="bf16",
)

# Profile before compiling — FlopCounterMode may not trace compiled graphs
optimizer.profile()
model = torch.compile(model)

for batch in dataloader:
    with optimizer.track_step() as result:  # calls zero_grad() at block entry
        loss = model(**batch).loss
        loss.backward()
    optimizer.step()                        # outside the timing window

    if step % 10 == 0:
        print(f"MFU {result.mfu:.3f}  MBU {result.mbu:.3f}")
```

### Context manager (bare PyTorch)

```python
from mfu_tracker import track, profile_flops, param_bytes

# Profile once on the uncompiled model before training begins
sample = {"input_ids": batch["input_ids"][:1]}
flops = profile_flops(model, kwargs=sample, with_backward=True)
p_bytes = param_bytes(model)

for batch in dataloader:
    optimizer.zero_grad()
    with track(flops, p_bytes, dtype="bf16") as result:
        loss = model(**batch).loss
        loss.backward()
    optimizer.step()

    print(f"MFU: {result.mfu:.3f}  MBU: {result.mbu:.3f}  {result.elapsed_sec*1000:.0f} ms/step")
```

---

## FLOP counting

```python
from mfu_tracker import profile_flops, flash_attn_flops, param_bytes

# Standard models — FlopCounterMode counts SDPA automatically on CUDA
flops = profile_flops(model, kwargs=batch, with_backward=True)

# flash_attn 2.5+ (Tri Dao's standalone PyPI package) is also counted
# automatically — it registers its kernels via torch.library.custom_op,
# so FlopCounterMode sees them. The manual flash_attn_flops() helper is only
# needed for older flash_attn versions (≤2.4) that bypass the dispatcher:
# flops += flash_attn_flops(batch_size=B, seq_len=S, num_heads=H, head_dim=D)

# PEFT / LoRA — restrict param_bytes to trainable parameters only
p_bytes = param_bytes(model, trainable_only=True)
```

`with_backward=True` applies the standard algorithmic 3× convention (1× forward + 2× backward). This is the *MFU* convention and does not change under gradient checkpointing — recomputation is a memory optimization, not part of the model's math.

For HFU, pass `hfu_backward_factor` to `MFUOptimizerWrapper`. The HF callback auto-detects from `TrainingArguments.gradient_checkpointing`, so you typically don't need to set it manually — pass an explicit value only for selective layer checkpointing.

| Setup | `hfu_backward_factor` | HFU total multiplier |
|-------|----------------------|---------------------|
| No checkpointing (default) | `2.0` | `3×` (matches MFU) |
| Full activation checkpointing | `3.0` | `4×` (Megatron-LM convention: HFU/MFU = 4/3) |
| Selective layer checkpointing | between `2.0` and `3.0` | `3.x×` |

The total multiplier is `1 + hfu_backward_factor`. The forward is 1× by definition; the rest is "backward + recomputation work" expressed as a multiple of forward FLOPs.

### MFU vs HFU

`profile_flops` and the published-MFU convention count the unmasked matmul work for `F.scaled_dot_product_attention`, regardless of `is_causal`. Hardware FLOPs Utilization (HFU) instead halves causal SDPA contributions to reflect what a flash-attention kernel actually executes (it skips the upper triangle). The `MFUOptimizerWrapper` and `MFUCallback` populate both metrics automatically when the model uses `is_causal=True`.

For bare `track()`, opt in by supplying both counts:

```python
from mfu_tracker import profile_flops_with_hfu, track

profile = profile_flops_with_hfu(model, kwargs=batch, with_backward=True)
with track(profile.flops, p_bytes, hfu_flop_count=profile.hfu_flops, dtype="bf16") as r:
    ...
print(f"MFU={r.mfu:.3f}  HFU={r.hfu:.3f}")
```

`HFU < MFU` only when the model uses `is_causal=True` on SDPA; for bidirectional models the two are equal in the no-checkpointing case. With gradient checkpointing, HFU > MFU because the GPU is doing extra recomputation work (set `hfu_backward_factor=3.0` for full activation checkpointing — see the table above). The HF callback adds `throughput/hfu` alongside `throughput/mfu` whenever the two would differ — either because of causal SDPA, gradient checkpointing, or both.

---

## GPU spec

```python
from mfu_tracker import get_gpu_spec

spec = get_gpu_spec()
print(spec.name)                        # e.g. "NVIDIA GeForce RTX 4080"
print(spec.peak_tflops("fp16"))         # e.g. 97.6
print(spec.peak_tflops("fp8"))          # Ada Lovelace (CC 8.9) and Hopper (CC 9.0)+
print(spec.peak_memory_bandwidth_tbs)   # e.g. 0.717
```

Supported dtypes: `fp32`, `fp16`, `bf16`, `int8`, `fp8`, `int4`, `fp4`. Unrecognized compute capabilities fall back to the nearest known major version with a `UserWarning`.

### How the peak numbers are derived

Rather than hard-coding the marketing TFLOPS value of every GPU model, the library derives peaks from first principles using only what `torch.cuda.get_device_properties()` reports plus a single tunable per architecture. The mental model:

```
fp16_peak_FLOPS = num_SMs × fp16_FLOPS_per_SM_per_clock × clock_rate_Hz
peak_for(dtype) = fp16_peak_FLOPS × dtype_multiplier
```

`num_SMs` and `clock_rate` come straight from the driver. `dtype_multiplier` reflects how many smaller values fit into the same tensor-core slot — half the bits means twice the throughput. The only architecture-specific number is **`fp16_FLOPS_per_SM_per_clock`** — how many FP16 tensor-core ops one SM completes per cycle:

| Architecture | (CC) | FP16 FLOPS/SM/clock |
|---|---|---|
| Volta | 7.0 | 1024 |
| Turing | 7.5 | 1024 |
| Ampere data-centre (A100) | 8.0 | 1024 |
| Ampere consumer (RTX 30xx) | 8.6 | 512 |
| Ada Lovelace (RTX 40xx) | 8.9 | 2048 |
| Hopper (H100) | 9.0 | 3840 |
| Blackwell (B100/B200) | 10.0 | 7680 |

The full table lives in [src/mfu_tracker/gpu.py](src/mfu_tracker/gpu.py) under `_FP16_FLOPS_PER_SM_PER_CLOCK` and `_DTYPE_MULTIPLIER` — adding a new architecture is a single dict entry.

Memory bandwidth is similarly derived: `memory_clock × bus_width × 2` (for DDR), again straight from device properties.

---

## Benchmark (RTX 4080 16 GB, GPT-2 medium 355M, bf16, seq=512)

| Configuration | MFU | HFU | MBU | tok/s | peak GB |
|---|---|---|---|---|---|
| batch=1 · eager | 4.4% | 4.4% | 1.5% | 7.3K | 3.68 |
| batch=4 · eager | 10.6% | 10.6% | 0.9% | 18.1K | 7.22 |
| batch=8 · eager | 10.6% | 10.6% | 0.4% | 18.1K | 13.72 |
| batch=8 · sdpa | 13.9% | 13.4% | 0.6% | 23.8K | 12.52 |
| batch=8 · sdpa + compile | 18.6% | 18.0% | 0.8% | 31.9K | 7.13 |
| batch=8 · flash_attention_2 | 13.9% | 13.4% | 0.6% | 23.8K | 11.10 |
| batch=8 · flash_attention_2 + compile | 20.0% | 19.3% | 0.8% | 34.3K | 7.17 |
| batch=8 · sdpa + ckpt | 14.0% | 18.1% | 0.6% | 24.1K | 10.69 |
| batch=8 · sdpa + compile + ckpt | 18.8% | 24.3% | 0.8% | 32.3K | 6.32 |
| batch=16 · sdpa + compile + ckpt | 18.9% | 24.3% | 0.4% | 32.4K | 11.94 |
| batch=16 · flash_attention_2 + compile + ckpt | 20.3% | 26.2% | 0.4% | 34.8K | 11.94 |

Reproduce: `python examples/benchmark_mfu.py` (defaults to gpt2-medium / bf16 / seq=512 on a 16 GB card; flash_attention_2 rows appear automatically if the `flash_attn` package is installed).

**Key observations:**

- **MFU caps at ~20% on this hardware.** GPT-2-medium isn't compute-heavy enough per kernel to saturate the 4080's tensor cores. 
- **`torch.compile` is the single biggest lever** — +5pp MFU and ~40% memory reduction (12.5G → 7.1G). 
- **`sdpa` and `flash_attention_2` are nearly identical without compile** (13.9% / 23.8K tok/s each) with the memory usage of `flash_attention_2` being slightly lower. 
- **Activation checkpointing has near-zero wall-time cost on modern GPUs** at this model and hardware scale. HFU rises to ~`4/3 × MFU` (the Megatron convention) because the GPU genuinely does ~33% more hardware work; MFU and tok/s barely move.
- **Bigger batch isn't buying throughput at this model size.** batch=16 at 32.4K tok/s vs batch=8 at 32.3K.Bigger batches matter for training dynamics (gradient noise, effective batch), not for raw efficiency on this hardware.
- **MBU is consistently ≤1%** — the model is not memory-bandwidth-bound which is expected during offline training. The combination of "low MFU + low MBU" suggests a kernel-launch / non-matmul-overhead bottleneck. `nvidia-smi` would show near 100% utilization on every row above which is uninformative.

---

## Multi-GPU

Leave `num_gpus=1` (the default) when using `profile_flops` as the FLOP source. For data-parallel strategies (DDP, FSDP), per-GPU FLOPs equal total FLOPs divided by N and wall time is the same on all ranks, so per-GPU MFU equals global MFU and the N factors cancel. Set `num_gpus > 1` only when pairing an analytically-derived full-model FLOP count (e.g. `6 × params × tokens`) with a total-job peak ceiling.

---

## Limitations

- **SDPA on CPU is not counted** — `FlopCounterMode` does not intercept flash attention dispatch on CPU. Profile with a CUDA model.
- **MFU counts causal attention unmasked** — `profile_flops` and `compute_mfu` follow the PaLM convention and report the raw matmul FLOPs for `F.scaled_dot_product_attention` regardless of `is_causal`. A causal flash-attention kernel actually executes ~half those FLOPs. The library reports HFU separately (see "MFU vs HFU" above) for the kernel-aware view. HFU detection only fires for `is_causal=True`; explicit causal masks via `attn_mask=`, sliding-window attention, and direct `flash_attn_func` calls are not auto-detected.
- **bitsandbytes quantized layers** — INT8/NF4 kernels are opaque to `FlopCounterMode`. NF4 dequantizes to fp16 before the matmul, so FLOP counts are approximately correct. Pass the appropriate dtype to use the right peak ceiling.
- **`flash_attn_func` direct calls** — models bypassing `F.scaled_dot_product_attention` need a manual `flash_attn_flops()` correction (see above).
- **MBU is a proxy** — the formula uses parameter bytes as a stand-in for memory traffic; actual DRAM traffic (activations, gradients, optimizer state) is higher and not measured.
- I have not tested the library extensively yet on larger GPUs or parallel configurations; please open an issue if you encounter unexpected behavior.

---

## Requirements

- Python 3.9+
- PyTorch 2.1+ (for `FlopCounterMode`)
- An Nvidia GPU

---

## License

MIT
