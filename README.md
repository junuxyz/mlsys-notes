# mlsys-notes

Learning notes and experiments for understanding modern Machine Learning System.

Currently focusing on LLM serving system and inference optimization.

## Notes
- [Introduction to LLM Inference Part 1](notes/llm-inference-intro-p1.md)
- [ORCA paper review](notes/orca.md)
- [PagedAttention paper review](notes/pagedattention.md)
- [Softmax: From Naive to Blocked Softmax](notes/softmax.md)

### Inference Engine
- [tinyorca deep dive](notes/tinyorca.md)
- [Inside nano-vLLM](notes/vllm/inside-nano-vllm.md)
- [How Multiprocess Serving Works in vLLM](notes/vllm/how-mp-serving-works-in-vllm.md)
### Distributed
- [Sarathi-Serve paper review](notes/sarathi-serve.md)
- [Tensor Parallelism](notes/distributed/tp.md)
- [Pipeline Parallelism](notes/distributed/pp.md)

### Hardware
- [NVIDIA GPU Architecture: From GPC to SM](notes/accelerators/gpu-architecture.md)
- [GPU Memory Hierarchy in CUDA](notes/accelerators/gpu-memory-hierarchy.md)

### Diffusion
- [Accelerating Diffusion Inference via Caching](notes/accelerating-diffusion-inference-via-caching.md)

## Labs
- [microengine](labs/microengine/README.md): a minimal serving engine
- [tinyorca](https://github.com/junuxyz/tinyorca): a minimal implementation of [ORCA](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [tiny-speculators](https://github.com/junuxyz/tiny-speculators): a from-scratch implementation of speculative decoding model training.
