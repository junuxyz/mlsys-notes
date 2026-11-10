# mlsys-notes

Learning notes and experiments for understanding modern Machine Learning System.

Currently focusing on LLM serving system and inference optimization.

## Notes
- [Introduction to LLM Inference Part 1](notes/llm-inference-intro-p1.md)
- [ORCA paper review](notes/orca.md)
- [PagedAttention paper review](notes/pagedattention.md)
- [Softmax: From Naive to Blocked Softmax](notes/softmax.md)
- [Rejection Sampling in Speculative Decoding](notes/speculative-decoding-rejection-sampling.md)
- [Triton Notes](labs/triton-notes/README.md)

### Transformer

- [MLA (Multi-Head Latent Attention) Explained](notes/transformer/mla.md)

### Inference Engine
- [tinyorca deep dive](notes/tinyorca.md)
- [Inside nano-vLLM](notes/inference-engine/inside-nano-vllm.md)
- [Inside nano-vLLM v1](notes/inference-engine/inside-nano-vllm-v1.md)
- [How Multiprocess Serving Works in vLLM](notes/vllm/how-mp-serving-works-in-vllm.md)
- [Comparing Round-Robin Routing vs KV-Aware Routing on Nemotron-3-Nano](https://github.com/Prasannajaga/deployment-guide/blob/main/blogs/rr-vs-kv-aware-routing.md)

### Distributed
- [Sarathi-Serve paper review](notes/sarathi-serve.md)
- [Tensor Parallelism](notes/distributed/tp.md)
- [Pipeline Parallelism](notes/distributed/pp.md)
- [Recovering NVLS on H100](notes/distributed/recovering-nvls-on-h100.md)

### Hardware
- [NVIDIA GPU Architecture: From GPC to SM](notes/accelerators/gpu-architecture.md)
- [GPU Memory Hierarchy in CUDA](notes/accelerators/gpu-memory-hierarchy.md)
- [CUDA Intro](notes/accelerators//cuda-intro.md)
- [[KR] NPU Memory Architecture and Data Reuse](notes/accelerators/kr_npu_memory_arch_and_data_reuse.pdf)


### Diffusion
- [Accelerating Diffusion Inference via Caching](notes/accelerating-diffusion-inference-via-caching.md)

## Labs
- [Triton notes and kernels](labs/triton-notes/README.md): GPU kernel experiments, notes, and interactive diagrams
- [microengine](labs/microengine/README.md): a minimal serving engine
- [tinyorca](https://github.com/junuxyz/tinyorca): a minimal implementation of [ORCA](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [nano-vllm-bench](https://github.com/junuxyz/nano-vllm-bench): a performance comparison between nano-vLLM and nano-vLLM-v1
- [tiny-speculators](https://github.com/junuxyz/tiny-speculators): a from-scratch implementation of speculative decoding model training.
- [Dynamo Deployment Guide](https://github.com/Prasannajaga/deployment-guide): deployed and benchmarked models including Nemotron Nano and Qwen3 on up to 16× H100 GPUs with experiments spanning P/D disaggregation, KV-aware routing, KV offloading, and replayed multi-turn coding-agent workloads.
- [MLA Lab](labs/mla-lab/README.md): roofline analysis and H100 experiments comparing MLA weight absorption methods
