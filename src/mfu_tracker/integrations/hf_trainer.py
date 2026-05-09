"""HuggingFace Trainer integration via TrainerCallback."""
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import TrainerCallback

from ..flops import param_bytes, profile_flops_with_hfu
from ..gpu import GPUSpec, get_gpu_spec
from ..tracker import compute_mbu, compute_mfu


class MFUCallback(TrainerCallback):
    """
    TrainerCallback that logs MFU, HFU (when applicable), and MBU at every
    Trainer logging step.

    MFU follows the algorithmic convention: ``fwd_flops × 3`` (forward + 2×
    backward). Gradient checkpointing does not affect MFU's numerator —
    recomputation is a memory optimization, not part of the math. HFU instead
    uses ``fwd_hfu_flops × (1 + hfu_backward_factor)``. Default ``2.0`` matches
    MFU (no recomputation). Pass ``3.0`` under full activation checkpointing
    (Megatron-LM convention: HFU/MFU = 4/3) so HFU reflects the actual FLOPs
    executed by the hardware.

    Per-step cost is two non-blocking ``Event.record()`` calls (~10 μs CPU, no
    GPU stall). The single ``torch.cuda.synchronize()`` is deferred to ``on_log``
    and amortised across all steps in the logging interval.

    Usage::

        from mfu_tracker.integrations.hf_trainer import MFUCallback

        # Zero-config: sample batch grabbed from train_dataloader, hfu_backward_factor
        # auto-detected from args.gradient_checkpointing.
        trainer = Trainer(..., callbacks=[MFUCallback(dtype="bf16")])

    **torch.compile**: profile_flops is called at ``on_train_begin``, before the
    first compiled step. This is correct — ``torch.compile`` does not change the
    FLOP count (same math, just faster execution). The MFU improvement from
    compilation is captured automatically in the CUDA event timing of real steps.
    Do not pass a compiled model to this callback directly; let HF Trainer compile
    after the callback is registered.

    **DDP / FSDP**: leave ``num_gpus=1`` — per-GPU MFU equals global MFU for
    data-parallel jobs.

    Args:
        sample_batch:    A representative batch dict passed as ``**kwargs`` to the
                         model. Used once at training start to profile forward FLOPs.
                         Pass ``None`` (default) to grab the first batch from
                         ``train_dataloader`` automatically at ``on_train_begin``.
                         For ``IterableDataset``, the auto-grabbed batch is consumed
                         and not seen by the training loop — pass an explicit batch
                         if you need to keep every sample.
        dtype:           Compute dtype for the peak ceiling — "fp16", "bf16", etc.
        num_gpus:        GPUs in the peak ceiling (default 1).
        hfu_backward_factor: Backward + recomputation cost as a multiple of
                         forward FLOPs. ``None`` (default) auto-detects from
                         ``args.gradient_checkpointing``: ``2.0`` if disabled,
                         ``3.0`` if enabled (full activation checkpointing,
                         Megatron-LM convention). Pass an explicit float to
                         override — useful for selective layer checkpointing
                         (values between 2.0 and 3.0). Does not affect MFU,
                         which always uses the algorithmic 3× multiplier.
        metric_prefix:   Prefix for logged metric names (default ``"throughput"``).
                         Results in ``throughput/mfu`` and ``throughput/mbu``, which
                         WandB groups into its own section away from loss/lr. Set to
                         ``""`` to log bare ``mfu`` / ``mbu`` keys.
        device:          CUDA device to query. Defaults to current device.
    """

    def __init__(
        self,
        sample_batch: Optional[dict[str, Any]] = None,
        dtype: str = "bf16",
        num_gpus: int = 1,
        hfu_backward_factor: Optional[float] = None,
        metric_prefix: str = "throughput",
        device: Optional[torch.device] = None,
    ) -> None:
        self.sample_batch = sample_batch
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.hfu_backward_factor = hfu_backward_factor
        self.metric_prefix = metric_prefix
        self.device = device

        self._model: Optional[nn.Module] = None
        self._spec: Optional[GPUSpec] = None
        self._fwd_flops: Optional[int] = None
        self._fwd_hfu_flops: Optional[int] = None
        self._param_bytes: Optional[int] = None

        # Each entry: (e_start, e_bwd, e_end, bwd_recorded)
        # Each entry: (e_start, e_end). Accumulated between on_log calls.
        self._pending: list[tuple] = []

    def _profile(self, model: nn.Module) -> None:
        assert self.sample_batch is not None  # caller ensures
        self._spec = get_gpu_spec(self.device)
        self._param_bytes = param_bytes(model)
        try:
            # Move sample batch to the model's device (Trainer may have moved the model).
            model_device = next(model.parameters()).device
            batch = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in self.sample_batch.items()
            }
            profile = profile_flops_with_hfu(
                model, kwargs=batch, with_backward=False
            )
            self._fwd_flops = profile.flops
            self._fwd_hfu_flops = profile.hfu_flops
        except Exception as exc:
            warnings.warn(
                f"mfu-tracker: FLOP profiling failed ({exc}); MFU will not be logged.",
                stacklevel=2,
            )

    # --- TrainerCallback protocol -------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.hfu_backward_factor is None:
            self.hfu_backward_factor = 3.0 if getattr(args, "gradient_checkpointing", False) else 2.0
        if self.sample_batch is None:
            train_dataloader = kwargs.get("train_dataloader")
            if train_dataloader is not None:
                try:
                    batch = next(iter(train_dataloader))
                    if isinstance(batch, dict):
                        self.sample_batch = batch
                    else:
                        warnings.warn(
                            "mfu-tracker: train_dataloader yielded a non-dict batch "
                            f"({type(batch).__name__}); pass sample_batch= explicitly.",
                            stacklevel=2,
                        )
                except Exception as exc:
                    warnings.warn(
                        f"mfu-tracker: could not auto-grab sample batch ({exc}); "
                        "pass sample_batch= explicitly.",
                        stacklevel=2,
                    )
        if model is not None and torch.cuda.is_available() and self.sample_batch is not None:
            self._model = model
            self._profile(model)

    def on_step_begin(self, args, state, control, **kwargs):
        if self._fwd_flops is None or not torch.cuda.is_available():
            return

        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
        self._pending_step = (e_start, e_end)

    def on_step_end(self, args, state, control, **kwargs):
        if not hasattr(self, "_pending_step"):
            return

        e_start, e_end = self._pending_step
        e_end.record()
        self._pending.append((e_start, e_end))
        del self._pending_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self._pending or self._fwd_flops is None or self._param_bytes is None:
            return

        # Single sync amortised across all accumulated steps.
        torch.cuda.synchronize(self.device)

        n_steps = len(self._pending)
        total_ms = sum(e_start.elapsed_time(e_end) for e_start, e_end in self._pending)
        self._pending.clear()

        elapsed_sec = total_ms / 1000
        if elapsed_sec <= 0:
            return

        # MFU uses the algorithmic 3× convention — independent of recomputation.
        total_flops = int(self._fwd_flops * n_steps * 3)
        # HFU uses the user-set or auto-detected factor; default to 2.0 if
        # on_train_begin never resolved it (e.g. direct on_log calls in tests).
        hfu_factor = self.hfu_backward_factor if self.hfu_backward_factor is not None else 2.0
        hfu_step_factor = n_steps * (1 + hfu_factor)

        prefix = f"{self.metric_prefix}/" if self.metric_prefix else ""
        logs[f"{prefix}mfu"] = round(
            compute_mfu(total_flops, elapsed_sec, dtype=self.dtype, num_gpus=self.num_gpus, spec=self._spec), 4
        )
        # Emit HFU when it is meaningfully different from MFU — either because the
        # model uses causal SDPA (fwd_hfu_flops < fwd_flops) or because gradient
        # checkpointing inflates real backward FLOPs (hfu_factor > 2.0).
        fwd_hfu = self._fwd_hfu_flops
        if fwd_hfu is not None and (fwd_hfu != self._fwd_flops or hfu_factor != 2.0):
            total_hfu_flops = int(fwd_hfu * hfu_step_factor)
            logs[f"{prefix}hfu"] = round(
                compute_mfu(total_hfu_flops, elapsed_sec, dtype=self.dtype, num_gpus=self.num_gpus, spec=self._spec), 4
            )
        logs[f"{prefix}mbu"] = round(
            compute_mbu(self._param_bytes * n_steps, elapsed_sec, num_gpus=self.num_gpus, spec=self._spec), 4
        )
