# mfu-tracker

PyPI library for tracking Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU).

## Architecture

- [src/mfu_tracker/gpu.py](src/mfu_tracker/gpu.py) — queries `torch.cuda.get_device_properties()` to derive peak TFLOPS and memory bandwidth from first principles. Uses `_FP16_FLOPS_PER_SM_PER_CLOCK` keyed by `(major, minor)` compute capability tuple (empirically validated against spec sheets). Supports per-dtype peak ceilings (fp16, bf16, int8, fp8, int4, fp4).
- [src/mfu_tracker/flops.py](src/mfu_tracker/flops.py) — FLOP counting via `torch.utils.flop_counter.FlopCounterMode` (PyTorch 2.1+ required). `FlopCounterMode` hooks at the ATen dispatch level, so it counts `F.scaled_dot_product_attention` (SDPA / native flash attention) automatically on CUDA — no manual correction needed for modern transformer models. SDPA is NOT counted on CPU (the kernel dispatches differently); profile with a CUDA model for accurate counts. If `FlopCounterMode` fails to trace (rare — custom autograd functions, opaque CUDA kernels), `profile_flops()` raises `RuntimeError` directing users to compute FLOPs analytically and pass them to `compute_mfu()` / `track()` directly. Tri Dao's `flash_attn` PyPI package (used by HF's `attn_implementation="flash_attention_2"`) registers its kernels via `torch.library.custom_op` since 2.5, so `_flash_attn_forward` is also intercepted automatically when the package is importable — exact count, not estimation. The `flash_attn_flops(B, S, H, D)` helper remains as a manual escape hatch for older flash_attn versions or rare cases that bypass the dispatcher. For kwargs-only models (e.g. HF), wrapping happens via `_KwargsAdapter`. `param_bytes()` accepts `trainable_only=True` for PEFT/LoRA backward MBU estimates. `profile_flops_with_hfu()` returns a `FlopProfile(flops, hfu_flops)` from a single trace — `hfu_flops` halves SDPA contributions where `is_causal=True`. Custom causal-aware formulas are registered into `FlopCounterMode` via `custom_mapping`, with a side-channel accumulator that tracks "savings" so the official total stays at the unmasked PaLM/MFU convention while HFU = total − savings. Only `is_causal=True` on the high-level SDPA aten ops and the regular `flash_attn::_flash_attn_forward` is detected; explicit causal masks via `attn_mask`, sliding-window attention, varlen flash_attn (`_flash_attn_varlen_forward` — packed sequences with different schema), and the lower-level `_flash_attention_forward` aten op fall back to default (unhalved) counting.
- [src/mfu_tracker/tracker.py](src/mfu_tracker/tracker.py) — `track()` context manager and `compute_mfu`/`compute_mbu` standalone functions. All accept a `dtype` parameter. `track()` accepts an optional `hfu_flop_count` kwarg; when supplied, `result.hfu` is populated alongside `result.mfu`. `UtilizationResult` is a mutable dataclass yielded by `track()`; fields start as `None` and are populated after the block exits. CUDA-event-backed fields (from `track_step()`) are resolved lazily on first attribute access; `_resolve()` is idempotent and calls `synchronize` exactly once. `result.hfu` stays `None` when the caller did not supply an HFU count (e.g. plain `track()` with one int).
- [src/mfu_tracker/optim.py](src/mfu_tracker/optim.py) — `MFUOptimizerWrapper`. Wraps any `torch.optim.Optimizer` and exposes a `track_step()` context manager. MFU's backward multiplier is hardcoded at 3× (algorithmic forward + 2× backward — Megatron-LM / PaLM convention; gradient checkpointing does not change MFU's numerator). HFU's multiplier is `1 + hfu_backward_factor` where the parameter is "backward + recomputation work as a multiple of forward FLOPs": 2.0 (default) = no recomputation, 3.0 = full activation checkpointing (2× matmul backward + 1× forward replay → 4× total, Megatron-LM HFU/MFU = 4/3). Higher values would imply double recomputation, which is not standard. No gradient hook. `zero_grad()` is called automatically at the **start** of `track_step()`; call `optimizer.step()` **after** the block to keep it outside the timing window. Profile the uncompiled model via `wrapper.profile()` before calling `torch.compile`.
- [src/mfu_tracker/integrations/hf_trainer.py](src/mfu_tracker/integrations/hf_trainer.py) — `MFUCallback(TrainerCallback)`. Zero-config in the typical case: `sample_batch` defaults to `None` and is auto-grabbed from `train_dataloader` (via `next(iter(...))`) at `on_train_begin`; `hfu_backward_factor` defaults to `None` and is auto-detected from `args.gradient_checkpointing` (3.0 if enabled, 2.0 if not). Both can be overridden with explicit values — useful for IterableDataset (so a batch isn't consumed for calibration) or selective layer checkpointing. Auto-grab gracefully degrades on non-dict batches or iterator failure (warns, skips MFU logging). Profiles the model once at `on_train_begin` (moves sample batch to model device automatically). Records two non-blocking CUDA events per step: `on_step_begin` records the start; `on_pre_optimizer_step` records the end (with `on_step_end` as a fallback for older HF versions). Ending at `on_pre_optimizer_step` excludes `optimizer.step()` from the timing window — its fp32 elementwise work would otherwise inflate elapsed time without contributing to the bf16/fp16 FLOP numerator (underestimating MFU by 2–5%). `torch.cuda.synchronize()` is deferred to `on_log`, amortised across the logging interval. Logs `throughput/mfu` and `throughput/mbu` (configurable prefix). Logs `throughput/hfu` whenever it would differ from MFU — either because the model uses causal SDPA, or because `hfu_backward_factor != 2.0` (gradient checkpointing). Does NOT read `state.total_flos` — HF Trainer uses the dense 6ND formula for all models including MoE, overcounting MoE by up to 4×. Skips silently when CUDA is unavailable.

## Key design decisions

- `(major, minor)` tuple keys for GPU lookup — CC 8.0 (A100) and CC 8.6 (RTX 3090) have genuinely different per-SM throughput (1024 vs 512 FP16 FLOPs/SM/clock) despite both being Ampere. Major-version-only keys would be wrong.
- Ada Lovelace is CC 8.9 — gets FP8 support via a special case in `_fp8_supported()` even though its major version is 8 (below the FP8 min_major of 9 for Hopper).
- **Fixed backward factor, not dynamic measurement.** The original design used a gradient hook on `trainable[-1]` to measure the forward/backward time split dynamically. This was abandoned because: (1) gradient hooks fire on the CPU autograd thread based on scheduling, not GPU completion — making the "start of backward" event unreliable; (2) with `optimizer.step()` inside the timing window, the measured factor absorbed optimizer time and gave ~4× instead of ~2×; (3) `torch.compile` restructures the backward graph so the hook often never fires, producing inconsistent FLOP multipliers between compiled and uncompiled runs. Fixed multiplier with a user-overridable parameter is simpler and consistent.
- **MFU vs HFU multiplier semantics.** MFU's multiplier is hardcoded at 3× (algorithmic: forward + 2× backward, the Megatron-LM / PaLM convention). It does not change under gradient checkpointing because recomputation is a memory optimization, not part of the model's math — checkpointing should *lower* MFU (less useful work per second) while leaving HFU stable. The user-tunable `hfu_backward_factor` (default 2.0) only feeds into HFU's `(1 + hfu_backward_factor)`; users set it to 3.0–4.0 under full activation checkpointing. This split was the whole point of distinguishing MFU from HFU; conflating them into a single `backward_factor` would defeat the purpose.
- `optimizer.step()` belongs **outside** `track_step()`. `zero_grad()` is called at the start of the block so gradients are valid until the block exits. This way timing captures forward + backward only.
- `UtilizationResult` is mutable (no `frozen=True`) so the context manager pattern works correctly — yield first, populate after block exits.
- `MFUOptimizerWrapper.track_step()` is lazy — no `synchronize` in `finally`, only on first attribute access of the result. Skipping `result.mfu` on some steps incurs zero sync cost for those steps.
- HF integration uses `TrainerCallback`, not monkey-patch — cleaner, composable, and avoids patching internal Trainer methods. `MFUCallback` inherits from `TrainerCallback` so HF Trainer can call all callback events on it.
- `metric_prefix="throughput"` on `MFUCallback` — logs `throughput/mfu` and `throughput/mbu`. WandB groups metrics by `/` separator, placing these in their own "throughput" section away from `loss`/`lr`. Set `metric_prefix=""` for bare keys.
- HF Trainer forwards the `logs` dict from `on_log` to all configured integrations (WandB, TensorBoard, MLflow) automatically — no extra configuration needed.
- Graceful degradation: unknown compute capability emits a `UserWarning` and falls back to the closest known major version.
- MBU is always reported alongside MFU.
- `num_gpus` parameter on `track()`, `compute_mfu()`, `compute_mbu()`, `MFUCallback`, and `MFUOptimizerWrapper` scales the peak ceiling. Default 1, correct for all parallelism strategies when using `profile_flops` — per-GPU MFU equals global MFU for DDP, FSDP, tensor, and pipeline parallelism because the N factors cancel (per-GPU FLOPs = total/N, wall time is the same across all GPUs). Only set `num_gpus > 1` when pairing analytically-derived full-model FLOPs (e.g. `6 × params × tokens`) with a total-job peak.
- `torch.compile` does not change FLOP count (same math, faster execution). Profile the *uncompiled* model — `FlopCounterMode` may not trace compiled graphs correctly. The MFU improvement from compilation is captured automatically via CUDA event timing of real steps.
- `src/` layout for correct PyPI packaging (hatchling build backend).

## Benchmark findings (RTX 4080, GPT-2 124M, fp16)

From `examples/benchmark_mfu.py` — uses `track()` with `profile_flops(with_backward=True)` (fixed 3× convention) for consistent results across all configurations:

| Configuration         | MFU   | ms/step |
|-----------------------|-------|---------|
| batch=1  \| eager     | ~2.7% | ~40ms   |
| batch=8  \| eager     | ~9%   | ~93ms   |
| batch=8  \| sdpa      | ~12%  | ~74ms   |
| batch=8  \| sdpa+compile | ~17% | ~50ms |
| batch=16 \| sdpa+compile | ~16% | ~104ms |

Key observations:
- Low MFU (~2–17%) is expected for GPT-2 on modern hardware — the model is too small to saturate tensor cores. Large models (LLaMA-70B) reach 40–60% MFU.
- Low MBU (~0.2–0.7%) at batch≥4 means memory bandwidth is **not** the bottleneck — the model is compute-bound (or kernel-launch-bound). MBU is more meaningful for inference at batch=1.
- Both low MFU and low MBU simultaneously indicates **kernel launch overhead**: the GPU idles between small operations waiting for the CPU to dispatch the next kernel. `torch.compile` addresses this by fusing kernels, giving +5–8pp MFU improvement.
- `sdpa` over `eager` attention: +2–3pp MFU from avoiding materialising the full B×H×S×S attention matrix (flash attention tiling).

## Testing

Two test tiers:

- **Mock-based** (`test_flops.py`, `test_gpu.py`, `test_tracker.py`, `test_hf_callback.py`, `test_optim.py`) — no GPU required, run anywhere.
- **GPU integration** (`test_integration_gpu.py`) — skipped automatically without CUDA. Validated on RTX 4080 (CC 8.9). Covers: spec detection without warnings, FLOP counts matching theory within 1% for `Linear` and `Conv2d`, MFU/MBU in `(0, 1]` on real hardware, `compute_mfu` agreeing with `track()`, larger batch → higher MFU.

Known faithfulness limitations:
- Peak ceiling is from NVIDIA spec sheets, not our own measurements.
- `F.scaled_dot_product_attention` (SDPA) is counted automatically on CUDA via `FlopCounterMode`. Models using `flash_attn_func` directly (rare — older HF with `use_flash_attention_2=True`) still need `flash_attn_flops()` correction.
- SDPA is not counted when profiling on CPU — profile the CUDA model for accurate counts.
- bitsandbytes INT8/NF4 quantized layers (QLoRA) are opaque CUDA kernels not visible to either counter. NF4 dequantizes to fp16 before matmul so FLOPs are approximately correct. Pass `dtype="int8"` to get the right peak ceiling.
- CUDA event timing is accurate; CPU-timer `track()` requires a `synchronize` at block boundaries.
- The dynamic backward factor measurement (gradient hook on `trainable[-1]`) was removed — it gave unreliable results (~4× instead of ~2×) due to CPU/GPU async timing and broke comparisons between compiled/uncompiled models.

`transformers` and `accelerate` are dev dependencies (needed to test `MFUCallback` and run the HF Trainer example).

```bash
uv sync --group dev
.venv/bin/pytest tests/ -v                          # mock tests only (no GPU needed)
.venv/bin/pytest tests/test_integration_gpu.py -v  # GPU tests
```

## Linting and type checking

After adding a feature or non-trivial change, run both linters and resolve all findings before considering the task done:

```bash
uvx ruff check .
uvx basedpyright
```

Config lives in `pyproject.toml` under `[tool.basedpyright]`:
- `typeCheckingMode = "standard"` — basedpyright defaults to `"all"`, stricter than pyright's `"strict"` and floods PyTorch code with noise.
- `reportPrivateImportUsage = "none"` — PyTorch's top-level surface (`torch.relu`, `torch.randn`, `torch.float16`, etc.) isn't formally re-exported in its stubs but is the documented API. Disabling avoids hundreds of false positives.
- `exclude = ["examples", ...]` — `GPT2Config(n_layer=..., n_head=...)` kwargs go through `**kwargs` in `transformers.PretrainedConfig` and aren't in the stubs. Type-checking examples produces noise without value.

Common patterns when basedpyright flags `Optional[X]` issues in `src/`:
- For invariants ("field A is None iff field B is None"), add `assert self.b is not None` after the early-return on `a`.
- For `Optional[float]` fields that get computed then re-read in the same method, compute via a local first then assign — basedpyright doesn't narrow attribute reassignment.
- For `torch.cuda.Event` and similar runtime-only opaque types, prefer `Any` over `Optional[object]` in dataclass fields.

## Examples

- `examples/benchmark_mfu.py` — benchmarks MFU/MBU across batch size, attention implementation (`eager` vs `sdpa`), and `torch.compile` using GPT-2 (124M). Uses `track()` with pre-profiled FLOPs for consistent results.
- `examples/hf_trainer_mfu.py` — demonstrates `MFUCallback` with HF Trainer on synthetic data. Metrics appear as `throughput/mfu` / `throughput/mbu` in training logs and WandB. Run with `--wandb` to enable WandB logging.
