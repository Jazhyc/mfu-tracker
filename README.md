# mfu-tracker

When profiling training runs, I found that most existing tools either lacked MFU/MBU support entirely or dragged in hundreds of megabytes of transitive dependencies. This library is an attempt at a self-contained alternative.

**mfu-tracker** is a PyTorch library for measuring Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU). It supports bare PyTorch training loops, an optimizer wrapper, and a HuggingFace Trainer callback.

- **Minimal dependencies** — PyTorch and `thop` only.
- **Profiled FLOPs, not formula estimates** — uses `FlopCounterMode` to count the FLOPs your model actually executes rather than a formula like `6 × params × tokens`. For Mixture-of-Experts models this means only active experts are counted, giving a more accurate numerator than parameter-based estimates.
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

If both MFU and MBU are low simultaneously, the GPU is underutilized. Two common causes: kernel dispatch overhead (the CPU cannot issue kernels fast enough to keep the GPU busy — `torch.compile` reduces this by fusing operations), or CPU-side pipeline stalls (slow DataLoader, heavy host preprocessing, or host-to-device transfers in the hot path).

---

## Installation

```bash
pip install mfu-tracker
```

HuggingFace Trainer integration requires no extra install — if you are already running HF Trainer, `transformers` is already available. Import `MFUCallback` directly.

---

## Usage

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

### HuggingFace Trainer

```python
from mfu_tracker.integrations.hf_trainer import MFUCallback

sample_batch = {k: v[:batch_size] for k, v in next(iter(train_dataloader)).items()}

callback = MFUCallback(
    sample_batch=sample_batch,
    dtype="bf16",
    metric_prefix="throughput",  # logs throughput/mfu and throughput/mbu
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[callback],
)
trainer.train()
```

`throughput/mfu` and `throughput/mbu` are added to the Trainer log dict at each logging step and forwarded automatically to any configured integrations (WandB, TensorBoard, MLflow). WandB groups metrics by the `/` separator, so these appear in a distinct "throughput" section rather than alongside loss and learning rate.

---

## FLOP counting

```python
from mfu_tracker import profile_flops, flash_attn_flops, param_bytes

# Standard models — FlopCounterMode counts SDPA automatically on CUDA
flops = profile_flops(model, kwargs=batch, with_backward=True)

# Models calling flash_attn_func directly (rare; older HF with use_flash_attention_2=True)
# need a manual correction since the C extension is opaque to FlopCounterMode:
flops += flash_attn_flops(batch_size=B, seq_len=S, num_heads=H, head_dim=D)

# PEFT / LoRA — restrict param_bytes to trainable parameters only
p_bytes = param_bytes(model, trainable_only=True)
```

`with_backward=True` applies the standard 3× convention (1× forward + 2× backward). For gradient checkpointing, pass `backward_factor=3.0` or `4.0` to `MFUOptimizerWrapper` or `MFUCallback`.

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

---

## Benchmark (RTX 4080, GPT-2 124M, fp16)

| Configuration | MFU | ms/step |
|---|---|---|
| batch=1 · eager | ~0.027 | ~40 ms |
| batch=8 · eager | ~0.09 | ~93 ms |
| batch=8 · sdpa | ~0.12 | ~74 ms |
| batch=8 · sdpa + compile | ~0.17 | ~50 ms |
| batch=16 · sdpa + compile | ~0.16 | ~104 ms |

GPT-2 (124M) is a small model relative to the compute capacity of a modern GPU, so low MFU is expected — the model spends a large fraction of step time waiting for kernel dispatch rather than doing arithmetic. Larger models (e.g. LLaMA-70B) typically reach 0.40–0.60 MFU. The improvement from `torch.compile` reflects kernel fusion reducing dispatch overhead. I'll add some testing on this later.

```bash
python examples/benchmark_mfu.py --help
python examples/hf_trainer_mfu.py --dtype bf16 --batch-size 16
```

---

## Multi-GPU

Leave `num_gpus=1` (the default) when using `profile_flops` as the FLOP source. For data-parallel strategies (DDP, FSDP), per-GPU FLOPs equal total FLOPs divided by N and wall time is the same on all ranks, so per-GPU MFU equals global MFU and the N factors cancel. Set `num_gpus > 1` only when pairing an analytically-derived full-model FLOP count (e.g. `6 × params × tokens`) with a total-job peak ceiling.

---

## Limitations

- **SDPA on CPU is not counted** — `FlopCounterMode` does not intercept flash attention dispatch on CPU. Profile with a CUDA model.
- **bitsandbytes quantized layers** — INT8/NF4 kernels are opaque to `FlopCounterMode`. NF4 dequantizes to fp16 before the matmul, so FLOP counts are approximately correct. Pass the appropriate dtype to use the right peak ceiling.
- **`flash_attn_func` direct calls** — models bypassing `F.scaled_dot_product_attention` need a manual `flash_attn_flops()` correction (see above).
- **Peak ceilings from spec sheets** — these are not independently measured. MFU > 1.0 indicates the ceiling is underestimated.
- **MBU is a proxy** — the formula uses parameter bytes as a stand-in for memory traffic; actual DRAM traffic (activations, gradients, optimizer state) is higher and not measured.
- I have not tested the library extensively yet; please open an issue if you encounter any bugs or unexpected behavior.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (2.1+ recommended for `FlopCounterMode`)
- A CUDA GPU is required for meaningful results; CPU timing works but MFU will be near zero for any realistic model

---

## License

MIT
