import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    # constexpr: this argument is not going to change a lot --
    # you can know it at compile time (static)
    BLOCK_SIZE: tl.constexpr,
):
    # each pids process different elements; SIMD
    # since it's 1d grid grid would be something
    # like (2,) if size = 2048
    PID = tl.program_id(axis=0)

    block_start = PID * BLOCK_SIZE
    # tl.arange creates vector of size (0 ~ blocksize)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load data from dram to sram
    # in triton, masked-off lanes still produce a tensor value
    # use a typed neutral value such as `other=0.0` for arithmetic kernels.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x + y

    # write data back to dram
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x, y):
    out = torch.empty_like(x)

    assert x.device == device and y.device == device
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    # grid must be tuple
    n_elements = out.numel()
    # cdiv(m, n) == (m + (n-1) // n)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']))

    # similar to CUDA's kernel<<<grid, block>>> just different syntax
    # triton internally creates meta dict which includes 'BLOCK_SIZE'
    # and uses grid(meta) internally
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

    return out

# atol: how much err can we tolerate in absolute terms
# rtol: for small nums
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=device):
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    # run triton kernel
    z_tri = add(x, y)
    # pytorch equivalent
    z_ref = x + y
    # compare
    # assert_close tells how off (far) away
    # it is if it exceeds the threshold
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("test passed")


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=1000000)
