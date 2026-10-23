"""Benchmark three MLA methods on a Modal H100 using BF16 and causal attention.

Run: modal run labs/mla-lab/measure.py
Batch=1, q_len=1/4096, kv_len=4096. Timing samples are saved under results/.
"""

import json
from pathlib import Path

import modal

Q_LENS = (1, 4096)
KV_LEN = 4096
HEAD = 128
METHODS = ("standard", "unmerged_absorb", "merged_absorb")
REPEATS = 5

app = modal.App("mla-lab")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-python==0.7.2", "cuda-python==12.9.4")
    .env({"MAX_JOBS": "4"})
)

with image.imports():
    import flashinfer
    import torch


def make_case(q_len, kv_len, method):
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(42)

    def weight(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) / shape[-2] ** 0.5

    # Hidden state (input for MLA)
    h = torch.randn(q_len, 5120, device=device, dtype=dtype)

    # Down projections (shared among heads)
    W_DQ = weight(5120, 1536)
    W_DKV = weight(5120, 512)
    W_KR = weight(5120, 64)

    # Up projections (per head)
    W_QR = weight(1536, HEAD * 64)
    W_UQ = weight(1536, HEAD * 128)
    W_UK = weight(HEAD, 512, 128)
    W_UV = weight(HEAD, 512, 128)
    W_O = weight(HEAD * 128, 5120)

    # KV cache
    c_KV = torch.randn(kv_len, 512, device=device, dtype=dtype)
    k_pe = torch.randn(kv_len, 64, device=device, dtype=dtype)

    q_norm = torch.ones(1536, device=device, dtype=dtype)
    kv_norm = torch.ones(512, device=device, dtype=dtype)
    positions = torch.arange(kv_len - q_len, kv_len, device=device, dtype=torch.int32)
    scale = (128 + 64) ** -0.5

    # 128MB memory for FlashInfer
    workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    # q pointer: shows where each sequence's query start and end
    qo_indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    # kv pointer: shows where each sequence's kv start and end
    kv_indptr = torch.tensor([0, kv_len], device=device, dtype=torch.int32)

    if method == "standard":
        # Combine the query projections and the KV projections before measurement.
        W_Q = torch.cat(
            (W_UQ.view(1536, HEAD, 128), W_QR.view(1536, HEAD, 64)), dim=-1
        ).reshape(1536, HEAD * (128 + 64))
        W_UKV = (
            torch.cat((W_UK, W_UV), dim=-1)
            .permute(1, 0, 2)
            .reshape(512, HEAD * (128 + 128))
        )
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="fa3"
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=HEAD,
            num_kv_heads=HEAD,
            head_dim_qk=128 + 64,
            head_dim_vo=128,
            causal=True,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
    else:
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=torch.arange(kv_len, device=device, dtype=torch.int32),
            kv_len_arr=torch.tensor([kv_len], device=device, dtype=torch.int32),
            num_heads=HEAD,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=True,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        if method == "merged_absorb":
            # Merge weights once in FP32, outside measurement, then store in BF16.
            W_UQUK = (
                torch.bmm(  # Per head
                    W_UQ.view(1536, HEAD, 128).permute(1, 0, 2).float(),
                    W_UK.transpose(1, 2).float(),
                )
                .permute(1, 0, 2)
                .reshape(1536, HEAD * 512)
                .to(dtype)
            )
            W_UV_O = (
                torch.bmm(W_UV.float(), W_O.view(HEAD, 128, 5120).float())
                .reshape(HEAD * 512, 5120)
                .to(dtype)
            )

    def prepare_inputs():
        c_Q = flashinfer.norm.rmsnorm(h @ W_DQ, q_norm, eps=1e-6)
        # Only new tokens are projected. Trials overwrite the same cache slice.
        c_KV_new = flashinfer.norm.rmsnorm(h @ W_DKV, kv_norm, eps=1e-6)
        k_pe_new = h @ W_KR
        c_KV[-q_len:].copy_(c_KV_new)
        k_pe[-q_len:].copy_(k_pe_new)
        return c_Q

    def apply_rope(q_pe):
        # Rotate queries and new keys at their absolute positions. Old keys are cached.
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q_pe, k_pe[-q_len:, None, :], positions, interleave=False
        )

    def standard():
        c_Q = prepare_inputs()
        Q = (c_Q @ W_Q).view(q_len, HEAD, 128 + 64)
        apply_rope(Q[..., 128:])
        # Only compressed KV is cached, so Standard must reconstruct all K/V each call.
        KV = (c_KV @ W_UKV).view(kv_len, HEAD, 128 + 128)
        K, V = KV.split(128, dim=-1)
        # Pack cached position keys with content keys for FA3 interface.
        # FA3 only accepts single fused vector per head.
        K = torch.cat((K, k_pe[:, None, :].expand(-1, HEAD, -1)), dim=-1)
        o_head = wrapper.run(Q, K, V)
        return o_head.reshape(q_len, HEAD * 128) @ W_O

    def unmerged_absorb():
        c_Q = prepare_inputs()
        q_pe = (c_Q @ W_QR).view(q_len, HEAD, 64)
        apply_rope(q_pe)
        q_nope = (c_Q @ W_UQ).view(q_len, HEAD, 128)
        q_absorbed = (
            torch.bmm(q_nope.transpose(0, 1), W_UK.transpose(1, 2))
            .transpose(0, 1)
            .contiguous()
        )
        # FlashInfer calls the absorbed query q_nope.
        o_latent = wrapper.run(
            q_nope=q_absorbed,
            q_pe=q_pe,
            ckv_cache=c_KV[:, None, :],
            kpe_cache=k_pe[:, None, :],
        )
        # Absorb W_UV to o_latent instead of reconstructing V
        o_head = torch.bmm(o_latent.transpose(0, 1), W_UV).transpose(0, 1).contiguous()
        return o_head.reshape(q_len, HEAD * 128) @ W_O

    def merged_absorb():
        c_Q = prepare_inputs()
        # We do not project q_nope
        q_pe = (c_Q @ W_QR).view(q_len, HEAD, 64)
        apply_rope(q_pe)
        q_absorbed = (c_Q @ W_UQUK).view(q_len, HEAD, 512)
        o_latent = wrapper.run(
            q_nope=q_absorbed,
            q_pe=q_pe,
            ckv_cache=c_KV[:, None, :],
            kpe_cache=k_pe[:, None, :],
        )
        return o_latent.reshape(q_len, HEAD * 512) @ W_UV_O

    def reference(query_indices):
        # Check selected queries against all KV tokens in FP32.
        c_Q = torch.nn.functional.rms_norm(
            h.float() @ W_DQ.float(), (1536,), q_norm.float(), 1e-6
        )
        c_KV_ref = c_KV.float().clone()
        c_KV_ref[-q_len:] = torch.nn.functional.rms_norm(
            h.float() @ W_DKV.float(), (512,), kv_norm.float(), 1e-6
        )
        k_pe_ref = k_pe.float().clone()
        inv_freq = 10000.0 ** (-torch.arange(0, 64, 2, device=device).float() / 64)
        angles = positions.float()[:, None] * inv_freq[None, :]
        cos, sin = angles.cos(), angles.sin()

        def rotate(x, cos, sin):
            left, right = x.chunk(2, dim=-1)
            return torch.cat(
                (left * cos - right * sin, right * cos + left * sin), dim=-1
            )

        k_pe_ref[-q_len:] = rotate(h.float() @ W_KR.float(), cos, sin)
        q_nope = (c_Q[query_indices].float() @ W_UQ.float()).view(-1, HEAD, 128)
        q_pe = (c_Q[query_indices].float() @ W_QR.float()).view(-1, HEAD, 64)
        q_pe = rotate(q_pe, cos[query_indices, None, :], sin[query_indices, None, :])
        K = (c_KV_ref @ W_UK.permute(1, 0, 2).reshape(512, HEAD * 128).float()).view(
            kv_len, HEAD, 128
        )
        scores_nope = torch.einsum("qhd,khd->qhk", q_nope, K)
        scores_pe = q_pe @ k_pe_ref.T
        # Bottom-right causal mask: decode sees all keys. Prefill skips future keys.
        q_positions = positions[query_indices]
        k_positions = torch.arange(kv_len, device=device)
        scores = (scores_pe + scores_nope) * scale
        scores.masked_fill_(
            k_positions[None, None, :] > q_positions[:, None, None], -torch.inf
        )
        attn = torch.softmax(scores, dim=-1)
        o_latent = attn @ c_KV_ref
        o_head = torch.einsum("qhc,hcd->qhd", o_latent, W_UV.float())
        return o_head.reshape(-1, HEAD * 128) @ W_O.float()

    return {
        "standard": standard,
        "unmerged_absorb": unmerged_absorb,
        "merged_absorb": merged_absorb,
    }[method], reference


@app.function(image=image, gpu="H100!", cpu=4, timeout=1800)
def benchmark():
    samples = []
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for q_len in Q_LENS:
            query_indices = list(range(min(4, q_len)))
            if q_len > 4:
                # Check 16 evenly spaced queries to reduce verification cost.
                query_indices = (
                    torch.linspace(0, q_len - 1, steps=min(16, q_len)).long().tolist()
                )
            cases = {method: make_case(q_len, KV_LEN, method) for method in METHODS}
            for method, (run, reference) in cases.items():
                print(f"Checking q_len={q_len}, method={method}", flush=True)
                actual = run()[query_indices].float()
                expected = reference(query_indices)
                assert actual.shape == expected.shape
                # BF16 projection rounding makes relative errors near zero misleading.
                relative_l2 = (actual - expected).norm(dim=-1) / expected.norm(dim=-1)
                print(
                    f"Max query relative L2 error: {relative_l2.max().item():.2%}",
                    flush=True,
                )
                torch.testing.assert_close(
                    relative_l2,
                    torch.zeros_like(relative_l2),
                    rtol=0,
                    atol=2e-2,
                    msg=lambda error, q=q_len, m=method: (
                        f"q_len={q}, method={m}\n{error}"
                    ),
                )
                for _ in range(10):
                    run()
            torch.cuda.synchronize()
            # Interleave methods to reduce bias from drift during the experiment.
            for trial in range(1, REPEATS + 1):
                for method, (run, _) in cases.items():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = run()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) * 1000
                    samples.append(
                        {
                            "q_len": q_len,
                            "method": method,
                            "trial": trial,
                            "time_us": elapsed,
                        }
                    )
                    print(
                        f"q_len={q_len}, {method}, trial={trial}: {elapsed:.1f} us",
                        flush=True,
                    )
                    del output
    return {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "q_lens": Q_LENS,
        "kv_len": KV_LEN,
        "methods": METHODS,
        "repeats": REPEATS,
        "dtype": "bf16",
        "causal": True,
        "scope": "h_to_output",
        "samples": samples,
    }


@app.local_entrypoint()
def main():
    data = benchmark.remote()
    # Overwrites the previous run.
    results = Path(__file__).with_name("results") / "modal"
    results.mkdir(parents=True, exist_ok=True)
    (results / "benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved benchmark: {results}")
