"""HuggingFace Trainer integration via monkey-patch."""
from __future__ import annotations

import warnings


def patch_trainer() -> None:
    """
    Monkey-patch transformers.Trainer to log MFU and MBU in its training loop.

    Call this before instantiating Trainer, or import mfu_tracker.integrations.hf
    which calls it automatically.

    No-ops gracefully if transformers is not installed.
    """
    try:
        from transformers import Trainer
    except ImportError:
        warnings.warn(
            "transformers is not installed; HF Trainer patch skipped.",
            stacklevel=2,
        )
        return

    if getattr(Trainer, "_mfu_tracker_patched", False):
        return

    # TODO: implement patch — wrap training_step to accumulate flop_count
    # and inject MFU/MBU into the logs dict passed to log()

    Trainer._mfu_tracker_patched = True
