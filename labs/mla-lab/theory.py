"""Compare matmul FLOPs and moved bytes for three MLA methods.

1. Standard: reconstruct K/V before attention.
2. Unmerged absorption: score compressed KV without constructing merged weights.
3. Merged absorption: use precomputed W_UQUK and W_UV_O.

Includes all matmuls from h to output.

For matmul [M, N] @ [N, K]:
- FLOPs = 2MNK
- bytes = 2(MN + NK + MK) for FP16/BF16.

We use the MLA dimensions from DeepSeek-V2.
"""

import argparse

HEAD = 128

# H100 SXM dense BF16 Tensor Cores and HBM3 bandwidth (without MFU consideration).
# https://www.nvidia.com/en-us/data-center/h100/
H100_FLOPS_PER_SECOND = 989.5e12
H100_BYTES_PER_SECOND = 3.35e12


def num_causal_pairs(q_len, kv_len):
    # Query i can attend to keys 0 through kv_len - q_len + i, inclusive.
    # Decode uses all kv_len keys. Full prefill uses kv_len * (kv_len + 1) / 2 pairs.
    return q_len * kv_len - q_len * (q_len - 1) // 2


# 1. Standard: reconstruct K/V before attention.
def num_flops_standard(q_len, kv_len):
    pairs = num_causal_pairs(q_len, kv_len)
    return 2 * (
        # h @ [W_DQ, W_DKV, W_KR] projects the new tokens.
        q_len * 5120 * (1536 + 512 + 64)
        # Build content query q_nope and position query q_pe from compressed c_Q.
        # c_Q [q_len * 1536] @ W_Q [1536 * (HEAD * (128 + 64))] = [q_len * (HEAD * (128 + 64))]
        + q_len * 1536 * (HEAD * (128 + 64))
        # Reconstruct content keys K and values V from the compressed KV cache.
        # c_KV [kv_len * 512] @ W_UKV [512 * (HEAD * (128 + 128))] = [kv_len * (HEAD * (128 + 128))]
        + kv_len * 512 * (HEAD * (128 + 128))
        + HEAD
        * (
            # Position scores: compare query and cached key position features.
            # q_pe [q_len * 64] @ k_pe.T [64 * kv_len] = [q_len * kv_len], per head
            pairs * 64
            # Content scores: compare content queries with reconstructed keys.
            # q_nope [q_len * 128] @ K.T [128 * kv_len] = [q_len * kv_len], per head
            + pairs * 128
            # o_head: sum reconstructed values using attn from both score contributions.
            # attn [q_len * kv_len] @ V [kv_len * 128] = [q_len * 128], per head
            + pairs * 128
        )
        # Concatenate the head outputs and project to hidden states.
        # concat(o_head) [q_len * (HEAD * 128)] @ W_O [(HEAD * 128) * 5120] = [q_len * 5120]
        + q_len * (HEAD * 128) * 5120
    )


def num_bytes_standard(q_len, kv_len):
    return 2 * (
        3 * q_len * 5120
        + 5120 * (1536 + 512 + 64)
        + q_len * (1536 + 512 + 64)
        + q_len * 1536
        + 1536 * (HEAD * (128 + 64))
        + q_len * (HEAD * (128 + 64))
        + kv_len * 512
        + 512 * (HEAD * (128 + 128))
        + kv_len * (HEAD * (128 + 128))
        + HEAD * (q_len * 64 + q_len * 128 + 128 * kv_len + kv_len * 128 + q_len * 128)
        # Shared k_pe reads across heads.
        + kv_len * 64
        + q_len * (HEAD * 128)
        + (HEAD * 128) * 5120
        + q_len * 5120
    )


# 2. Unmerged absorption: score compressed KV without constructing merged weights.
def num_flops_unmerged_absorb(q_len, kv_len):
    pairs = num_causal_pairs(q_len, kv_len)
    return 2 * (
        # h @ [W_DQ, W_DKV, W_KR] projects the new tokens.
        q_len * 5120 * (1536 + 512 + 64)
        # Build query position features q_pe from compressed c_Q.
        # c_Q [q_len * 1536] @ W_QR [1536 * (HEAD * 64)] = [q_len * (HEAD * 64)]
        + q_len * 1536 * (HEAD * 64)
        # Expand c_Q into content query features q_nope.
        # c_Q [q_len * 1536] @ W_UQ [1536 * (HEAD * 128)] = [q_len * (HEAD * 128)]
        + q_len * 1536 * (HEAD * 128)
        # Transform q_nope into q_absorbed to score compressed KV directly.
        # q_nope [q_len * 128] @ W_UK.T [128 * 512] = [q_len * 512], per head
        + HEAD * q_len * 128 * 512
        + HEAD
        * (
            # Position scores. q_pe contributes to attn through these scores.
            # q_pe [q_len * 64] @ k_pe.T [64 * kv_len] = [q_len * kv_len], per head
            pairs * 64
            # Content scores using c_KV as compressed keys.
            # q_absorbed [q_len * 512] @ c_KV.T [512 * kv_len] = [q_len * kv_len], per head
            + pairs * 512
            # o_latent: sum c_KV as compressed values, weighted by attn.
            # attn [q_len * kv_len] @ c_KV [kv_len * 512] = [q_len * 512], per head
            + pairs * 512
        )
        # Expand compressed output o_latent into the value output o_head.
        # o_latent [q_len * 512] @ W_UV [512 * 128] = [q_len * 128], per head
        + HEAD * q_len * 512 * 128
        # Concatenate head outputs and project to hidden states.
        # concat(o_head) [q_len * (HEAD * 128)] @ W_O [(HEAD * 128) * 5120] = [q_len * 5120]
        + q_len * (HEAD * 128) * 5120
    )


def num_bytes_unmerged_absorb(q_len, kv_len):
    return 2 * (
        3 * q_len * 5120
        + 5120 * (1536 + 512 + 64)
        + q_len * (1536 + 512 + 64)
        + q_len * 1536
        + 1536 * (HEAD * 64)
        + q_len * (HEAD * 64)
        + q_len * 1536
        + 1536 * (HEAD * 128)
        + q_len * (HEAD * 128)
        + HEAD
        * (
            q_len * 128
            + 128 * 512
            + q_len * 512
            + q_len * 64
            + q_len * 512
            + q_len * 512
            + q_len * 512
            + 512 * 128
            + q_len * 128
        )
        # Shared c_KV and k_pe reads across heads.
        + kv_len * (64 + 512)
        + q_len * (HEAD * 128)
        + (HEAD * 128) * 5120
        + q_len * 5120
    )


# 3. Merged absorption: use precomputed W_UQUK and W_UV_O.
def num_flops_merged_absorb(q_len, kv_len):
    pairs = num_causal_pairs(q_len, kv_len)
    return 2 * (
        # h @ [W_DQ, W_DKV, W_KR] projects the new tokens.
        q_len * 5120 * (1536 + 512 + 64)
        # Build query position features q_pe.
        # c_Q [q_len * 1536] @ W_QR [1536 * (HEAD * 64)] = [q_len * (HEAD * 64)]
        + q_len * 1536 * (HEAD * 64)
        # Build q_absorbed directly using merged query-key weights.
        # c_Q [q_len * 1536] @ W_UQUK [1536 * (HEAD * 512)] = [q_len * (HEAD * 512)]
        + q_len * 1536 * (HEAD * 512)
        + HEAD
        * (
            # Position scores. q_pe contributes to attn through these scores.
            # q_pe [q_len * 64] @ k_pe.T [64 * kv_len] = [q_len * kv_len], per head
            pairs * 64
            # Content scores using c_KV as compressed keys.
            # q_absorbed [q_len * 512] @ c_KV.T [512 * kv_len] = [q_len * kv_len], per head
            + pairs * 512
            # o_latent: sum c_KV as compressed values, weighted by attn.
            # attn [q_len * kv_len] @ c_KV [kv_len * 512] = [q_len * 512], per head
            + pairs * 512
        )
        # Apply merged value and output projections to concatenated compressed outputs.
        # concat(o_latent) [q_len * (HEAD * 512)] @ W_UV_O [(HEAD * 512) * 5120] = [q_len * 5120]
        + q_len * (HEAD * 512) * 5120
    )


def num_bytes_merged_absorb(q_len, kv_len):
    return 2 * (
        3 * q_len * 5120
        + 5120 * (1536 + 512 + 64)
        + q_len * (1536 + 512 + 64)
        + q_len * 1536
        + 1536 * (HEAD * 64)
        + q_len * (HEAD * 64)
        + q_len * 1536
        + 1536 * (HEAD * 512)
        + q_len * (HEAD * 512)
        + HEAD * (q_len * 64 + q_len * 512 + q_len * 512)
        # Shared c_KV and k_pe reads across heads.
        + kv_len * (64 + 512)
        + q_len * (HEAD * 512)
        + (HEAD * 512) * 5120
        + q_len * 5120
    )


def estimate_time(q_len, kv_len, method):
    flops = globals()[f"num_flops_{method}"](q_len, kv_len)
    traffic = globals()[f"num_bytes_{method}"](q_len, kv_len)
    compute_ms = flops / H100_FLOPS_PER_SECOND * 1000
    memory_ms = traffic / H100_BYTES_PER_SECOND * 1000
    return {
        "flops": flops,
        "bytes": traffic,
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
        "estimate_ms": max(compute_ms, memory_ms),
        # Assume we can fully overlap compute and memory
        "bound": "compute" if compute_ms >= memory_ms else "memory",
    }


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-len", type=positive_int, default=1)
    parser.add_argument("--kv-len", type=positive_int, default=4096)
    args = parser.parse_args()

    if args.q_len > args.kv_len:
        raise ValueError("q_len must be smaller or same as kv_len.")

    print(f"H100 SXM, dense BF16, causal, q_len={args.q_len}, kv_len={args.kv_len}")
    print(
        f"{'Method':<18} {'Compute ms':>12} {'Memory ms':>12} {'Ideal ms':>12} {'Model bound':>12}"
    )
    for method in ("standard", "unmerged_absorb", "merged_absorb"):
        result = estimate_time(args.q_len, args.kv_len, method)
        print(
            f"{method:<18} {result['compute_ms']:>12.3f} {result['memory_ms']:>12.3f} {result['estimate_ms']:>12.3f} {result['bound']:>12}"
        )


if __name__ == "__main__":
    main()
