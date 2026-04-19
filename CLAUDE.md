# mfu-tracker

PyPI library for tracking Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU).

## Architecture

- [src/mfu_tracker/gpu.py](src/mfu_tracker/gpu.py) — queries `torch.cuda.get_device_properties()` to derive peak TFLOPS and memory bandwidth from first principles. Uses `_FP16_FLOPS_PER_SM_PER_CLOCK` keyed by `(major, minor)` compute capability tuple (empirically validated against spec sheets). Supports per-dtype peak ceilings (fp16, bf16, int8, fp8, int4, fp4).
- [src/mfu_tracker/flops.py](src/mfu_tracker/flops.py) — FLOP counting via `thop`. Wraps `thop.profile()` which instruments actual torch ops via hooks — correctly counts only activated expert FLOPs in MoE models. For kwargs-only models (e.g. HF), wraps in a `_KwargsAdapter` since thop calls `model(*inputs)` with no kwargs support.
- [src/mfu_tracker/tracker.py](src/mfu_tracker/tracker.py) — `track()` context manager and `compute_mfu`/`compute_mbu` standalone functions. All accept a `dtype` parameter. `UtilizationResult` is a mutable dataclass yielded by `track()`; fields start as `None` and are populated after the block exits. Note: `return value` inside `@contextmanager` is silently swallowed by `contextlib` — the correct pattern is to yield the result object and mutate it after the block.
- [src/mfu_tracker/optim.py](src/mfu_tracker/optim.py) — `MFUOptimizerWrapper`. Wraps any `torch.optim.Optimizer` and exposes a `track_step()` context manager. Backward factor is measured dynamically: a gradient hook on `trainable[-1]` (last parameter in forward order = first to receive gradients in backward) fires a CUDA event at the start of backward; `backward_factor = bwd_ms / fwd_ms` is derived from CUDA event timings. Gradient checkpointing and other recomputation effects are captured automatically. `zero_grad()` is called automatically on block exit.
- [src/mfu_tracker/integrations/hf_trainer.py](src/mfu_tracker/integrations/hf_trainer.py) — `MFUCallback(TrainerCallback)`. Profiles the model once at `on_train_begin` with a user-supplied sample batch, then measures wall time per step to log `mfu` and `mbu` at each logging interval. Does NOT read `state.total_flos` — HF Trainer uses the dense 6ND formula for all models including MoE, overcounting MoE by up to 4×.

## Key design decisions

- `(major, minor)` tuple keys for GPU lookup — CC 8.0 (A100) and CC 8.6 (RTX 3090) have genuinely different per-SM throughput (1024 vs 512 FP16 FLOPs/SM/clock) despite both being Ampere. Major-version-only keys would be wrong.
- Ada Lovelace is CC 8.9 — gets FP8 support via a special case in `_fp8_supported()` even though its major version is 8 (below the FP8 min_major of 9 for Hopper).
- `thop` over `calflops` — calflops unconditionally imports `transformers` in `__init__.py`, making it a 600MB transitive dep that defeats the lightweight goal.
- Dynamic backward factor via gradient hook — avoids requiring users to know about gradient checkpointing or set a `backward_factor` manually. Hooking `trainable[-1]` works because backward traverses parameters in reverse forward order.
- `UtilizationResult` is mutable (no `frozen=True`) so the context manager pattern works correctly — yield first, populate after block exits.
- HF integration uses `TrainerCallback`, not monkey-patch — cleaner, composable, and avoids patching internal Trainer methods.
- Graceful degradation: unknown compute capability emits a `UserWarning` and falls back to the closest known major version.
- MBU is always reported alongside MFU.
- `src/` layout for correct PyPI packaging (hatchling build backend).

## Dev setup

```bash
uv add --dev pytest pytest-cov
.venv/bin/pytest tests/ -v
```
