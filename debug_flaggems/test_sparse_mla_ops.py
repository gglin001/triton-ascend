import pytest
import torch

from sparse_mla import triton_sparse_mla_fwd_interface
from sparse_mla_fwd import ref_sparse_mla_fwd_interface
from accuracy_utils import gems_assert_close, init_seed, to_reference

import triton
import npu_device_cpu  # noqa
# import torch_npu  # noqa

DEVICE = triton.runtime.driver.active.get_active_torch_device()
device = DEVICE


def make_sparse_mla_input(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
):
    """Create input data for sparse MLA operator"""
    init_seed(42)
    B = batch_size
    S = seq_len_q
    H = num_heads
    DQK = qk_dim
    SKV = seq_len_kv
    HKV = num_kv_heads

    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device).requires_grad_(
        requires_grad
    )
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
        requires_grad
    )

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    return q, kv, indices


def reference_sparse_mla_implementation(q, kv, indices, sm_scale=None, d_v=512):
    """Reference implementation - using provided reference function"""
    return ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale, d_v=d_v)


@pytest.mark.sparse_mla_forward
@pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("seq_len_q", [64, 128, 512])
@pytest.mark.parametrize("seq_len_q", [64])
# @pytest.mark.parametrize("seq_len_kv", [1024, 2048, 4096])
@pytest.mark.parametrize("seq_len_kv", [1024])
# @pytest.mark.parametrize("num_heads", [64, 128, 256])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("qk_dim", [576])  # Your operator is fixed at 576
@pytest.mark.parametrize("d_v", [512])  # Output dimension
# @pytest.mark.parametrize("topk", [64, 128, 256])
@pytest.mark.parametrize("topk", [64])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_sparse_mla_forward(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    num_kv_heads: int,
    qk_dim: int,
    d_v: int,
    topk: int,
    dtype: torch.dtype,
):
    """Sparse MLA forward propagation test"""
    # Skip unsupported cases
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    if topk > seq_len_kv:
        pytest.skip("topk cannot be larger than seq_len_kv")

    # Create input
    q, kv, indices = make_sparse_mla_input(
        batch_size,
        seq_len_q,
        seq_len_kv,
        num_heads,
        num_kv_heads,
        qk_dim,
        topk,
        dtype,
        device,
    )

    # Reference implementation
    ref_q = to_reference(q, False)
    ref_kv = to_reference(kv, False)
    ref_indices = to_reference(indices, False)

    ref_output = reference_sparse_mla_implementation(
        ref_q, ref_kv, ref_indices, d_v=d_v
    )

    # Your operator implementation
    your_output, your_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=d_v)

    # Accuracy comparison
    gems_assert_close(your_output, ref_output, dtype, atol=1e-2)
