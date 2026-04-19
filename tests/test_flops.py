from mfu_tracker.flops import transformer_flops


def test_training_flops():
    # 6 * N * D for training
    assert transformer_flops(1_000_000, 1024, with_backward=True) == 6 * 1_000_000 * 1024


def test_inference_flops():
    # 2 * N * D for inference
    assert transformer_flops(1_000_000, 1024, with_backward=False) == 2 * 1_000_000 * 1024
