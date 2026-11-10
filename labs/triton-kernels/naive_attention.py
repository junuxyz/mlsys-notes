"""Naive attention

Tensor layout:
    Q, K, V: [batch, heads, sequence, head_dim]
    S, P:    [batch, heads, sequence, sequence]
    O:       [batch, heads, sequence, head_dim]

The point is:
we launch three separate Triton kernels so S and P are
materialized in HBM between kernel launches.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _qk_kernel(
    q_ptr, k_ptr, s_ptr,
    n_heads, seq_len, head_dim,
    stride_qb, stride_qh, stride_qn, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_sb, stride_sh, stride_sm, stride_sn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Q @ K^T -> S.
    Q: M, K
    K: K, N
    S: M, N
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch = pid_bh // n_heads
    head = pid_bh % n_heads

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    m_mask = offs_m < seq_len
    n_mask = offs_n < seq_len

    q_ptrs = (
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_k[None, :] * stride_qk
    )

    k_ptrs = (
        k_ptr
        + batch * stride_kb
        + head * stride_kh
        # transpose here
        + offs_k[:, None] * stride_kk
        + offs_n[None, :] * stride_kn
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_block in tl.range(tl.cdiv(head_dim, BK)):
        k_block_mask = k_block * BK + offs_k < head_dim
        q_mask = m_mask[:,None] & k_block_mask[None,:]
        k_mask = k_block_mask[:,None] & n_mask[None,:]

        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc = tl.dot(q, k, acc=acc)

        q_ptrs += BK * stride_qk
        k_ptrs += BK * stride_kk
        
    acc = acc.to(tl.float16)

    s_ptrs = (
        s_ptr
        + batch * stride_sb
        + head * stride_sh
        + offs_m[:,None] * stride_sm
        + offs_n[None,:] * stride_sn
    )
    s_mask = m_mask[:,None] & n_mask[None,:]

    tl.store(s_ptrs, acc, mask=s_mask)


@triton.jit
def _softmax_kernel(
    s_ptr, p_ptr, scale,
    n_heads, seq_len,
    stride_sb, stride_sh, stride_sm, stride_sn,
    stride_pb, stride_ph, stride_pm, stride_pn,
    BN: tl.constexpr,
):
    """softmax(S * scale, dim=-1) -> P.
    S: MN
    P: N
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // n_heads
    head = pid_bh % n_heads

    offs_n = tl.arange(0, BN)
    n_mask = offs_n < seq_len

    s_ptrs = (
        s_ptr
        + batch * stride_sb
        + head * stride_sh
        + pid_m * stride_sm
        + offs_n * stride_sn
    )

    s = tl.load(s_ptrs, mask=n_mask, other=-float("inf"))
    s *= scale
    s -= tl.max(s, axis=0)
    num = tl.exp(s)
    denom = tl.sum(num, axis=0)
    p = num / denom

    p_ptrs = (
        p_ptr
        + batch * stride_pb
        + head * stride_ph
        + pid_m * stride_pm
        + offs_n * stride_pn
    )

    tl.store(p_ptrs, p, mask=n_mask)


@triton.jit
def _pv_kernel(
    p_ptr, v_ptr, o_ptr,
    n_heads, seq_len, head_dim,
    stride_pb, stride_ph, stride_pm, stride_pk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """P @ V -> O
    P: M, K
    V: K, N
    O: M, N
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch = pid_bh // n_heads
    head = pid_bh % n_heads

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    m_mask = offs_m < seq_len
    n_mask = offs_n < head_dim

    p_ptrs = (
        p_ptr
        + batch * stride_pb
        + head * stride_ph
        + offs_m[:,None] * stride_pm
        + offs_k[None,:] * stride_pk
    )
    v_ptrs = (
        v_ptr
        + batch * stride_vb
        + head * stride_vh
        + offs_k[:,None] * stride_vk
        + offs_n[None,:] * stride_vn
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_block in tl.range(tl.cdiv(seq_len, BK)):
        k_block_mask = k_block * BK + tl.arange(0, BK) < seq_len
        p_mask = m_mask[:,None] & k_block_mask[None,:]
        v_mask = k_block_mask[:,None] & n_mask[None,:]

        p = tl.load(p_ptrs, mask=p_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(p, v, acc=acc)

        p_ptrs += BK * stride_pk
        v_ptrs += BK * stride_vk

    o = acc.to(tl.float16)
    o_ptrs = (
        o_ptr
        + batch * stride_ob
        + head * stride_oh
        + offs_m[:,None] * stride_om
        + offs_n[None,:] * stride_on
    )
    o_mask = m_mask[:,None] & n_mask[None,:]

    tl.store(o_ptrs, o, mask=o_mask)


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """Naive self-attention forward pass"""
    assert q.ndim == k.ndim == v.ndim == 4, "all tensor must have (batch, n_heads, seq_len, head_dim)"
    assert q.shape == k.shape == v.shape, "all shape must be the same"
    assert q.device == k.device == v.device
    assert q.is_cuda, "Q, K, and V must be CUDA tensors"
    assert q.dtype == k.dtype == v.dtype

    batch, n_heads, seq_len, head_dim = q.shape

    # These two quadratic intermediates are intentionally allocated in **HBM**
    # we would want to avoid this in a more efficient algorithm
    s = torch.empty(
        (batch, n_heads, seq_len, seq_len),
        device=q.device,
        dtype=q.dtype,
    )
    p = torch.empty_like(s)
    o = torch.empty_like(q)
    
    qk_grid = lambda meta: (
        triton.cdiv(seq_len, meta["BM"]), # M of tile S[MxN]
        triton.cdiv(seq_len, meta["BN"]), # N of tile S[MxN]
        batch * n_heads,
    )

    softmax_grid = (
        seq_len,
        batch * n_heads,
    )

    pv_grid = lambda meta: (
        triton.cdiv(seq_len, meta["BM"]), # M of tile O[MxN]
        triton.cdiv(head_dim, meta["BN"]), # N of tile O[MxN]
        batch * n_heads,
    )

    _qk_kernel[qk_grid](
        q, k, s,
        n_heads, seq_len, head_dim,
        # k has same shape as q
        *q.stride(), *k.stride(), *s.stride(),
        BM=64, BN=64, BK=32,
    )

    _softmax_kernel[softmax_grid](
        s, p,
        head_dim**-0.5, # scale
        n_heads, seq_len,
        *s.stride(), *p.stride(),
        BN = triton.next_power_of_2(seq_len)
    )

    _pv_kernel[pv_grid](
        p, v, o,
        n_heads, seq_len, head_dim,
        *p.stride(), *v.stride(), *o.stride(),
        BM=64, BN=32, BK=64,
    )

    return o


def test_naive_attention(
    q_size: tuple[int, int, int, int],
    k_size: tuple[int, int, int, int],
    atol=1e-3,
    rtol=1e-3,
    ):
    torch.manual_seed(0)
    q = torch.rand(q_size, device=DEVICE, dtype=torch.float16)
    k = torch.rand(k_size, device=DEVICE, dtype=torch.float16)
    batch, n_heads, seq_len, head_dim = q.shape
    v = torch.rand((batch, n_heads, seq_len, head_dim), device=DEVICE, dtype=torch.float16)

    tri_o = naive_attention(q, k, v)
    ref_o = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=0.0,
        is_causal=False,
    )

    torch.testing.assert_close(tri_o, ref_o, atol=atol, rtol=rtol)
    print("PASSED")


configs = [
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128, 256, 512, 1024, 2048],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch SDPA", "Naive Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Latency (ms)",
        plot_name="naive-attention-performance",
        args={
            "batch": 1,
            "n_heads": 1,
            "head_dim": 128,
        },
    )
]


@triton.testing.perf_report(configs)
def benchmark(seq_len, batch, n_heads, head_dim, provider):
    shape = (batch, n_heads, seq_len, head_dim)

    q = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    k = torch.randn(shape, device=DEVICE, dtype=torch.float16)
    v = torch.randn(shape, device=DEVICE, dtype=torch.float16)

    if provider == "torch":
        fn = lambda: F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        fn = lambda: naive_attention(q, k, v)

    p50_ms, p05_ms, p95_ms = triton.testing.do_bench(
        fn, quantiles=[0.5, 0.05, 0.95],
    )
    
    def tflops(ms):
        flops = 4 * batch * n_heads * seq_len**2 * head_dim
        return flops * 1e-12 / (ms * 1e-3)
    return tflops(p50_ms), tflops(p05_ms), tflops(p95_ms)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(print_data=True, show_plots=True, save_path='.')
    else:
        test_naive_attention(
            (1, 1, 512, 128),
            (1, 1, 512, 128),
        )
