# mfu-tracker

PyPI library for tracking Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU).

## Architecture

- [src/mfu_tracker/gpu.py](src/mfu_tracker/gpu.py) — queries `torch.cuda.get_device_properties()` to derive peak TFLOPS and memory bandwidth from first principles. Uses `_FP16_FLOPS_PER_SM_PER_CLOCK` keyed by `(major, minor)` compute capability tuple (empirically validated against spec sheets). Supports per-dtype peak ceilings (fp16, bf16, int8, fp8, int4, fp4).
- [src/mfu_tracker/flops.py](src/mfu_tracker/flops.py) — FLOP counting via `torch.utils.flop_counter.FlopCounterMode` (PyTorch 2.1+), with `thop` as fallback. `FlopCounterMode` hooks at the ATen dispatch level, so it counts `F.scaled_dot_product_attention` (SDPA / native flash attention) automatically on CUDA — no manual correction needed for modern transformer models. SDPA is NOT counted on CPU (the kernel dispatches differently); profile with a CUDA model for accurate counts. `thop` fallback handles PyTorch < 2.1. For the rare `flash_attn` C extension (direct `flash_attn_func` calls), use `flash_attn_flops(B, S, H, D)` to compute the missing FLOPs manually. For kwargs-only models (e.g. HF), both paths wrap in `_KwargsAdapter`. `param_bytes()` accepts `trainable_only=True` for PEFT/LoRA backward MBU estimates.
- [src/mfu_tracker/tracker.py](src/mfu_tracker/tracker.py) — `track()` context manager and `compute_mfu`/`compute_mbu` standalone functions. All accept a `dtype` parameter. `UtilizationResult` is a mutable dataclass yielded by `track()`; fields start as `None` and are populated after the block exits. Note: `return value` inside `@contextmanager` is silently swallowed by `contextlib` — the correct pattern is to yield the result object and mutate it after the block.
- [src/mfu_tracker/optim.py](src/mfu_tracker/optim.py) — `MFUOptimizerWrapper`. Wraps any `torch.optim.Optimizer` and exposes a `track_step()` context manager. Backward factor is measured dynamically: a gradient hook on `trainable[-1]` (last parameter in forward order = first to receive gradients in backward) fires a CUDA event at the start of backward; `backward_factor = bwd_ms / fwd_ms` is derived from CUDA event timings. Gradient checkpointing and other recomputation effects are captured automatically. `zero_grad()` is called automatically on block exit.
- [src/mfu_tracker/integrations/hf_trainer.py](src/mfu_tracker/integrations/hf_trainer.py) — `MFUCallback(TrainerCallback)`. Profiles the model once at `on_train_begin` with a user-supplied sample batch, then measures wall time per step to log `mfu` and `mbu` at each logging interval. Does NOT read `state.total_flos` — HF Trainer uses the dense 6ND formula for all models including MoE, overcounting MoE by up to 4×. Uses same gradient hook + CUDA event technique as `MFUOptimizerWrapper` but defers `torch.cuda.synchronize()` to `on_log`, so the per-step GPU overhead is three non-blocking `Event.record()` calls only.

## Key design decisions

- `(major, minor)` tuple keys for GPU lookup — CC 8.0 (A100) and CC 8.6 (RTX 3090) have genuinely different per-SM throughput (1024 vs 512 FP16 FLOPs/SM/clock) despite both being Ampere. Major-version-only keys would be wrong.
- Ada Lovelace is CC 8.9 — gets FP8 support via a special case in `_fp8_supported()` even though its major version is 8 (below the FP8 min_major of 9 for Hopper).
- `thop` over `calflops` — calflops unconditionally imports `transformers` in `__init__.py`, making it a 600MB transitive dep that defeats the lightweight goal.
- Dynamic backward factor via gradient hook — avoids requiring users to know about gradient checkpointing or set a `backward_factor` manually. Hooking `trainable[-1]` works because backward traverses parameters in reverse forward order.
- `UtilizationResult` is mutable (no `frozen=True`) so the context manager pattern works correctly — yield first, populate after block exits. Fields from `track_step()` (CUDA-event-backed) are lazily resolved on first attribute access; `_resolve()` is idempotent and calls `synchronize` exactly once.
- `MFUOptimizerWrapper.track_step()` is also lazy — no `synchronize` in `finally`, only on first attribute access of the result. This means skipping `result.mfu` on some steps incurs zero sync cost for those steps.
- HF integration uses `TrainerCallback`, not monkey-patch — cleaner, composable, and avoids patching internal Trainer methods.
- Graceful degradation: unknown compute capability emits a `UserWarning` and falls back to the closest known major version.
- MBU is always reported alongside MFU.
- `num_gpus` parameter on `track()`, `compute_mfu()`, `compute_mbu()`, `MFUCallback`, and `MFUOptimizerWrapper` scales the peak ceiling. Default 1. **DDP / FSDP: leave at 1.** `profile_flops` returns per-GPU FLOPs; per-GPU MFU = global MFU for data-parallel jobs (the N factors cancel in numerator and denominator). **Tensor / pipeline parallelism**: each GPU runs 1/N of the model, so `profile_flops` on a sharded model undercounts by N. Supply full-model FLOPs analytically and set `num_gpus=N`.
- `torch.compile` does not change FLOP count (same math, faster execution). Profile the *uncompiled* model — `FlopCounterMode` may not trace compiled graphs correctly. The MFU improvement from compilation is captured automatically via CUDA event timing of real steps.
- `src/` layout for correct PyPI packaging (hatchling build backend).

## Testing

Two test tiers:

- **Mock-based** (`test_flops.py`, `test_gpu.py`, `test_tracker.py`, `test_hf_callback.py`, `test_optim.py`) — no GPU required, run anywhere.
- **GPU integration** (`test_integration_gpu.py`) — skipped automatically without CUDA. Validated on RTX 4080 (CC 8.9). Covers: spec detection without warnings, thop FLOP counts matching theory within 1% for `Linear` and `Conv2d`, MFU/MBU in `(0, 1]` on real hardware, `compute_mfu` agreeing with `track()`, larger batch → higher MFU.

Known faithfulness limitations:
- Peak ceiling is from NVIDIA spec sheets, not our own measurements.
- `F.scaled_dot_product_attention` (SDPA) is counted automatically on CUDA via `FlopCounterMode`. Models using `flash_attn_func` directly (rare — older HF with `use_flash_attention_2=True`) still need `flash_attn_flops()` correction.
- SDPA is not counted when profiling on CPU — profile the CUDA model for accurate counts.
- bitsandbytes INT8/NF4 quantized layers (QLoRA) are opaque CUDA kernels not visible to either counter. NF4 dequantizes to fp16 before matmul so FLOPs are approximately correct. Pass `dtype="int8"` to get the right peak ceiling.
- CUDA event timing is accurate; CPU-timer `track()` requires a `synchronize` at block boundaries.

`transformers` is a dev dependency (needed to test `MFUCallback` without installing the `hf` optional extra).

```bash
uv sync --group dev
.venv/bin/pytest tests/ -v                          # mock tests only (no GPU needed)
.venv/bin/pytest tests/test_integration_gpu.py -v  # GPU tests
```
