import torch
import random
import numpy as np


# Accuracy check function
def gems_assert_close(actual, expected, dtype, equal_nan=False, atol=None, rtol=None):
    # For bfloat16 and float16, use more relaxed tolerance
    if atol is None:
        if dtype in [torch.bfloat16, torch.float16]:
            atol = 1e-2
            rtol = 1e-2
        else:
            atol = 1e-4
            rtol = 1e-4

    print(f"Actual shape: {actual.shape}, Expected shape: {expected.shape}")
    print(f"Actual dtype: {actual.dtype}, Expected dtype: {expected.dtype}")

    # Calculate difference statistics
    diff = torch.abs(actual - expected)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(
        f"Actual range: [{torch.min(actual).item():.6f}, {torch.max(actual).item():.6f}]"
    )
    print(
        f"Expected range: [{torch.min(expected).item():.6f}, {torch.max(expected).item():.6f}]"
    )

    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan
    )


def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_reference(tensor, requires_grad=False):
    result = tensor.detach().clone()
    if requires_grad:
        result.requires_grad_()
    return result
