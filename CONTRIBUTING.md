# Contributing to mfu-tracker

Contributions are welcome — especially for adding new GPU architectures, validating MoE / multi-GPU setups, and fixing bugs.

## Setup

```bash
git clone https://github.com/<your-fork>/mfu-tracker.git
cd mfu-tracker
uv sync --group dev
```

`uv` resolves and installs everything (PyTorch, transformers, accelerate, pytest, etc.) into a local `.venv/`.

## Running tests

```bash
# Mock tests — no GPU required, run anywhere (~5 seconds)
uv run pytest tests/ --ignore=tests/test_integration_gpu.py

# GPU integration tests — skipped automatically without CUDA
uv run pytest tests/test_integration_gpu.py
```

Both must pass before opening a PR. Mock tests cover all the logic; the GPU tests validate spec detection and end-to-end MFU/MBU on real hardware.

## Linting and type checking

```bash
uvx ruff check .
uvx basedpyright
```

Both must show clean before merge. Config lives in `pyproject.toml` under `[tool.basedpyright]`. If you hit `Optional[X]` issues in `src/`, the patterns documented in [CLAUDE.md](CLAUDE.md) usually apply.

## Common contribution paths

### Adding a new GPU architecture

Edit [src/mfu_tracker/gpu.py](src/mfu_tracker/gpu.py):

1. Add an entry to `_FP16_FLOPS_PER_SM_PER_CLOCK` keyed by `(major, minor)` compute capability tuple.
2. Verify the value against NVIDIA's spec sheet using `num_SMs × per_SM_rate × clock_GHz`. The README explains the derivation.
3. If the architecture introduces a new dtype (e.g. FP4 in Blackwell), update `_DTYPE_MULTIPLIER` with the right `(min_major, multiplier)` entry.
4. Open an issue if you don't have access to the hardware to verify — we can sanity-check via spec-sheet math even without running on it.

### Adding a new dtype

Same file, `_DTYPE_MULTIPLIER` dict. The multiplier is "how many of these values fit per fp16 tensor-core slot" — see the GPU spec section of the README for the mental model.

### Multi-GPU / per-rank metrics

Currently the library reports per-rank MFU correctly under DDP/FSDP (the math works out so per-rank == global), but doesn't surface rank imbalance. A PR that adds per-rank logging in `MFUCallback` (e.g. via `torch.distributed.all_gather` of `tokens_per_sec` at log time) would be a meaningful improvement. Open an issue first to discuss the API.

### Bug fixes

Include a regression test in the appropriate file (`tests/test_*.py`) — most issues can be reproduced with mocks; only reach for `test_integration_gpu.py` if real CUDA behavior is involved.

## PR checklist

Before opening a PR:

- [ ] `uv run pytest tests/` passes (run the GPU tests too if you have CUDA)
- [ ] `uvx ruff check .` clean
- [ ] `uvx basedpyright` clean
- [ ] New features have tests
- [ ] If touching `src/mfu_tracker/gpu.py`, the math has been verified against an NVIDIA spec sheet
- [ ] README / CLAUDE.md updated if the change affects user-facing behavior or design decisions
