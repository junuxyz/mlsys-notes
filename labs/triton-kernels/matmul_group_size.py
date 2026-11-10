import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_M, a_stride_K,
    b_stride_K, b_stride_N,
    c_stride_M, c_stride_N,
    # meta params
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    # GROUP_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size
    
    block_offsets_M = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_offsets_N = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_offsets_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = block_offsets_M[:,None] * a_stride_M + block_offsets_K[None,:] * a_stride_K
    b_offsets = block_offsets_K[:,None] * b_stride_K + block_offsets_N[None,:] * b_stride_N
    
    m_mask = block_offsets_M < M
    n_mask = block_offsets_N < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_tile in range(0, num_pid_k):
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

def matmul(a, b, group_size=8):
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
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        num_warps=4,
        num_stages=4,
        GROUP_SIZE=group_size,
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
    print("matmul PASSED")

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["torch", 1, 2, 4, 8],
        line_names=[
            "PyTorch",
            "Group=1 (row-major)",
            "Group=2",
            "Group=4",
            "Group=8",
        ],
        styles=[
            ("green", "-"),
            ("black", "-"),
            ("blue", "-"),
            ("orange", "-"),
            ("red", "-"),
        ],
        ylabel="TFLOPS",
        plot_name="matmul-group-size-performance",
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, group_size=provider),
            quantiles=quantiles,
        )
    perf = lambda t: 2 * M * N * K * 1e-12 / (t * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def profile_group(group_size):
    M = N = K = 4096

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    matmul(a, b, group_size=group_size)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(20):
        matmul(a, b, group_size=group_size)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    import sys

    if "--profile-group" in sys.argv:
        index = sys.argv.index("--profile-group")
        group_size = int(sys.argv[index + 1])
        profile_group(group_size)
    
    else:
        test_matmul((512, 512))
        test_matmul((1025, 1025))

"""
Example usage:
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=group1 \
  uv run python matmul.py --profile-group 1
"""
