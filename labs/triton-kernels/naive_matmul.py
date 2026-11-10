import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2)
]

@triton.autotune(configs = autotune_configs, key=['M','N','K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_M, a_stride_K,
    b_stride_K, b_stride_N,
    c_stride_M, c_stride_N,
    # meta params
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_tile_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tile_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tile_k = tl.cdiv(K, BLOCK_SIZE_K)

    pid_m = pid // num_tile_n
    pid_n = pid % num_tile_n

    block_offsets_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_offsets_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = block_offsets_M[:,None] * a_stride_M + block_offsets_K[None,:] * a_stride_K
    b_offsets = block_offsets_K[:,None] * b_stride_K + block_offsets_N[None,:] * b_stride_N
    
    m_mask = block_offsets_M < M
    n_mask = block_offsets_N < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_tile in range(0, num_tile_k):
        k_mask = k_tile * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K
        a_mask = m_mask[:,None] & k_mask[None,:]
        b_mask = k_mask[:,None] & n_mask[None,:]

        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)

        a_offsets += BLOCK_SIZE_K * a_stride_K
        b_offsets += BLOCK_SIZE_K * b_stride_K

    c = acc.to(tl.float16)
    c_offsets = block_offsets_M[:,None] * c_stride_M + block_offsets_N[None,:] * c_stride_N
    c_mask = m_mask[:,None] & n_mask[None,:]
    tl.store(c_ptr + c_offsets, c, mask=c_mask)


# matmul
def matmul(a, b):
    assert a.device == b.device
    assert a.ndim == b.ndim == 2, "only support matrix"
    assert a.shape[1] == b.shape[0], "K should be same"
    assert a.dtype == b.dtype == torch.float16, "both must be float16"

    (M, K), (_, N) = a.shape, b.shape

    c = torch.empty((M,N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),) # 1d grid

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

# test kernel
def test_matmul(
    size: tuple[int, int],
    atol=1e-2,
    rtol=1e-2,
    device=DEVICE,
):
    torch.manual_seed(0)
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)

    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

# benchmark (reuse https://github.com/evintunador/triton_docs_tutorials/blob/main/06_matmul/matmul.py)
configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "naive-matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    test_matmul((512, 512))
    test_matmul((1025, 1025))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
