import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# def kernel
@triton.jit
def _add_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    z = x + y

    tl.store(z_ptr + offsets, z, mask=mask)

# kernel launch
def add(
    x: torch.Tensor,
    y: torch.Tensor,
    ):
    assert x.device == y.device == DEVICE, "must be in same device"
    assert x.numel() == y.numel(), "size must be same"

    z = torch.empty_like(x)
    n_elements = z.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](
        x, y, z,
        n_elements, BLOCK_SIZE=1024,
    )

    return z

# test
def test_vector_add(
    size: int,
    atol=1e-2,
    rtol=1e-2,
):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE, dtype=torch.float16)
    y = torch.rand(size, device=DEVICE, dtype=torch.float16)

    tri_z = add(x, y)
    ref_z = x + y

    torch.testing.assert_close(tri_z, ref_z, atol=atol, rtol=rtol)
    print("PASSED")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    ))

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_vector_add(512)
    test_vector_add(513)

    import sys
    if "--benchmark" in sys.argv:
        benchmark.run(print_data=True, show_plots=True)
