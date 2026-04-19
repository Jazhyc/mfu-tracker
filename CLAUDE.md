# mfu-tracker

PyPI library for tracking Model FLOPs Utilization (MFU) and Model Bandwidth Utilization (MBU).

## Architecture

- `src/mfu_tracker/gpu.py` — queries `torch.cuda.get_device_properties()` to derive peak TFLOPS and memory bandwidth from first principles (compute capability → tensor cores/SM → peak FLOPs). No per-SKU table; clock speed is queried live.
- `src/mfu_tracker/flops.py` — FLOP counting formulas (6ND transformer approximation + standalone interface).
- `src/mfu_tracker/tracker.py` — `track()` context manager and `compute_mfu`/`compute_mbu` standalone functions.
- `src/mfu_tracker/integrations/hf_trainer.py` — HuggingFace Trainer monkey-patch (call `patch_trainer()` before instantiating).

## Key design decisions

- Compute capability major version → tensor cores/SM via `_TENSOR_CORES_PER_SM` dict (only needs updating for new GPU generations).
- Graceful degradation: unknown compute capability emits a `UserWarning` and uses a conservative fallback.
- MBU is always reported alongside MFU.
- `src/` layout for correct PyPI packaging (hatchling build backend).

## Dev setup

```bash
pip install -e ".[dev]"
pytest
```
