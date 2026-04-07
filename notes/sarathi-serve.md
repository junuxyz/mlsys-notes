# Sarathi Serve

## TL;DR

> To explain Sarathi-Serve in one sentence, it introduces a way to overcome _generation stalls_ from decode requests by implementing chunked prefill and stall-free batching.
>
> We will talk more about what this means later.
>
> While the field seems to favor optimizing prefill and decode separately (P/D disaggregation), the ideas introduced in this paper<sup><a href="#reference-1">[1]</a></sup> are still widely used in industry standard inference engines such as vLLM or SGLang (especially for single node cases).

## Background

In the background section, I will mainly talk about four takeaways, following the paper's narrative.

I will not talk about batching, since I've already explained it in my previous notes.<sup><a href="#reference-11">[11]</a></sup><sup><a href="#reference-12">[12]</a></sup>

One thing to note is that batching works especially well for decode requests rather than prefill requests. This is because if the input is long enough, just one prefill request can saturate the GPU and make it compute-bound. Since it is already compute-bound, there is no merit in adding batches.

On the other hand, decode is usually memory-bound, so throughput scales linearly when requests are batched together. Again, I talked about this in Introduction to LLM Inference Part 1.<sup><a href="#reference-12">[12]</a></sup>, so I will not repeat myself.

The figure in the paper shows this:

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-01-prefill-vs-decode-batching.png"
    alt="Prefill and decode batching throughput comparison"
    width="600"
  />
  <br />
  <sub>Figure 1. Batching improves decode throughput much more than prefill throughput.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>


So the first takeaway is:

### Takeaway 1

> The two phases of LLM inference, prefill and decode, demonstrate contrasting behaviors wherein batching boosts decode phase throughput immensely but has little effect on prefill throughput.


Now let's talk about how to measure total execution time in LLM inference. More generally, for any program that does something repetitively, execution is either bound by the time spent computing the data or by the time spent loading and storing it in memory.

We can express this as $T = \max(T_{\text{math}}, T_{\text{mem}})$.

How can we ignore the non-dominant time cost? We will look at this with a simple example, borrowing Horace He's aesthetic image<sup><a href="#reference-2">[2]</a></sup>:

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-01-memory-bound-single-workload.png"
    alt="Memory-bound execution for a single workload"
    width="420"
  />
  <br />
  <sub>Figure 2. A memory-bound workload where memory transfer dominates compute for a single request.<sup><a href="#reference-2">[2]</a></sup></sub>
</p>

Suppose we have a memory-bound scenario (which resembles a decode scenario) with low arithmetic intensity.<sup><a href="#reference-12">[12]</a></sup> Computing the data takes only one second, while retrieving and rewriting the data takes 8 seconds.

If we run only one workload on such a program, the total time would be 1 + 8 = 9 seconds.

However, suppose we have more than 1,000 workloads repeating the same thing.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-02-memory-bound-batched-workloads.png"
    alt="Memory-bound execution with repeated batched workloads"
    width="480"
  />
  <br />
  <sub>Figure 3. Repeated memory-bound workloads hide compute time and remain dominated by memory access.<sup><a href="#reference-2">[2]</a></sup></sub>
</p>

When the first batch of data goes in, compute is done after a second, and then it stays idle for 7 seconds until the next batch of data comes in. So if multiple workloads are continuously executed, the 1s cost of compute gets hidden and the system is bound by the dominant time (in this case, $T_{\text{mem}}$) until $T_{\text{math}}$ exceeds 8 seconds. 

This is why $T$ can be expressed as $\max(T_{\text{mem}}, T_{\text{math}})$.

This also means that, theoretically, increasing the smaller cost adds no extra end-to-end latency until it reaches the dominant cost.

### Takeaway 2

> So the second takeaway we have is:
>
> Decode batches operate in a memory-bound regime leaving compute underutilized.
>
> This implies that more tokens can be processed along with a decode batch without significantly increasing its latency.



Now we will review earlier inference engines: FasterTransformer, Orca, and vLLM (v0). We will first compare FasterTransformer and vLLM v0, and then talk about Orca.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-03-vllm-vs-fastertransformer.png"
    alt="vLLM v0 and FasterTransformer scheduling comparison"
    width="620"
  />
  <br />
  <sub>Figure 4. vLLM v0 and FasterTransformer compared as prefill-prioritizing and decode-prioritizing schedulers.<sup><a href="#reference-3">[3]</a></sup></sub>
</p>

vLLM v0 (Option 1 in the image above) is _prefill-prioritizing_ and eagerly admits prefills before resuming ongoing decodes. As seen in the image, even if a new request comes in, it prioritizes the new request's prefill over the earlier requests' decode steps. This leads to high throughput because we can batch more decode steps after every request is in the decode phase, but it also leads to higher TPOT.

In the image, A and B need to _wait_ for C_p and D_p to finish processing. This is called _generation stalls_, which I mentioned in the [tl;dr](#tldr).

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-04-vllm-generation-stalls.png"
    alt="Generation stalls in vLLM caused by prefill prioritization"
    width="480"
  />
  <br />
  <sub>Figure 5. In vLLM, decode steps stall when newly arrived prefill requests are prioritized.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>


FasterTransformer (Option 2) used static batching, which takes a request batch, runs prefill and decode steps on it until all the requests are finished processing, then executes the next request batch in the same manner.

This can be called a _decode-prioritizing_ system, since it prioritizes decode steps instead of adding new prefills to increase throughput. This leads to low TPOT, but overall throughput is also low.

Orca was the only engine back then that supported mixed (hybrid) batching (in early 2024).

Orca introduced selective batching, which batches all non-attention operations and runs attention per request only (since attention should be applied per request). For an in-depth explanation, check out Improving LLM Inference with Continuous Batching: Orca through tinyorca.<sup><a href="#reference-11">[11]</a></sup>

However, Orca's hybrid batching did not turn out to be optimal because it increased latency, which often exceeded the SLO.

### What is SLO?

SLO stands for Service Level Objective, which refers to a target performance level for a particular metric. This sets a standard for what is considered acceptable service that does not harm user experience.<sup><a href="#reference-4">[4]</a></sup>

Usually, in the context of LLM inference, this boils down to metrics such as TPOT and TTFT. Throughout this post, let's assume TPOT must be <10ms.



<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-05-vllm-orca-latency-slo.png"
    alt="Decode latency exceeding the SLO in vLLM and Orca"
    width="360"
  />
  <br />
  <sub>Figure 6. Both vLLM and Orca exceed the decode latency SLO under mixed batching.<sup><a href="#reference-3">[3]</a></sup></sub>
</p>

Above is an image showing that both vLLM and Orca do not satisfy the latency SLO (10ms). We already saw why this happens in vLLM (it prioritizes prefill requests first). But you can see that Orca's hybrid batching increases latency in a fairly subtle manner.

### The effect of mixed batching

Hybrid batching works as follows:

1. Run token-wise operations (LayerNorm, QKV projection, MLP) on a flattened batch: `[sum_i S_i, hidden]`
2. Split into per-request tensors when execution reaches the attention path, where request `i` is `[S_i, hidden]`
3. **Run the request-local attention path independently per request.**
4. Re-flatten the outputs into a single token stream.
5. Continue with token-wise operations and repeat for the next layer.

In a hybrid iteration, decode-side attention work may finish earlier than the larger prefill-side attention work, but decode requests still wait for the completion of the entire hybrid iteration before their next token can be returned.<sup><a href="#reference-5">[5]</a></sup><sup><a href="#reference-6">[6]</a></sup>

Although prefill remains the dominant part of the hybrid iteration, the latency can slightly exceed that of a standalone prefill (as seen in the image above; 8ms vs 9ms) because the batch also includes decode-side work and incurs additional hybrid-batching overhead, such as batch formation and split/merge-related<sup><a href="#reference-7">[7]</a></sup> execution overhead.

So the step time for hybrid batching comes from $T_{\text{hybrid}} \approx T_{\text{batched non-attention}} + T_{\text{attention related operation for requests}} + T_{\text{overhead}}$.

Note: The attention-related term is not directly observable as a single per-request quantity; it emerges from the combined execution of the hybrid batch.

### Takeaway 3

> The third takeaway the authors mentioned is that the interleaving of prefills and decodes involves a trade-off between throughput and latency for current LLM inference schedulers.
>
> State-of-the-art systems (like vLLM v0) use prefill-prioritizing schedules, which improve throughput but also have side effects, most notably generation stalls.
>
> Another takeaway I want to explicitly mention (while already explained implicitly) is that not all step sizes are the same. Decode steps finish much more quickly, prefill steps take longer, and a hybrid batch of them takes even longer than that due to overhead.


### Pipeline Parallelism

Pipeline parallelism (PP) is a method where we shard the model weights vertically, dividing them into groups of layers. For example, if we use four GPUs for PP with a model that has 80 layers, each device will take 20 layers (e.g., device 0 gets layers 0-19, device 1 gets layers 20-39, ...). Unlike training, we do not need a backward pass for inference, so it is an even better fit for inference than for training.

So if we apply this to a forward pass scenario, GPU 0 processes the first 20 layers and sends data to GPU 1 and GPU 1 processes the next 20 layers, and so on.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-06-pipeline-parallelism-overview.png"
    alt="Pipeline parallelism overview across multiple GPUs"
    width="600"
  />
  <br />
  <sub>Figure 7. Pipeline parallelism splits layers across devices so one micro-batch flows stage by stage.<sup><a href="#reference-8">[8]</a></sup></sub>
</p>

For more in-depth explanation on PP, I recommend these two articles.<sup><a href="#reference-9">[9]</a></sup><sup><a href="#reference-10">[10]</a></sup>

### Pipeline Bubbles

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-07-pipeline-bubble.png"
    alt="Pipeline bubble caused by uneven step times"
    width="540"
  />
  <br />
  <sub>Figure 8. Uneven iteration times create pipeline bubbles where some devices wait idly.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

Another problem that appears when using multiple GPUs in Orca is pipeline bubbles.

In [Takeaway 3](#takeaway-3), I already mentioned that step sizes are not the same. Since the size of each step has high variance, performance is bound by the longest processing time among the GPUs at that moment.

For example, in the image above, GPU0 finishes processing the prefills for C and D while GPU1 is still processing the prefills for A and B, whose inputs are longer and therefore take more time. GPU0 has to wait until GPU1 is finished before it can pass data along. This gets even worse when GPU1 is processing prefill while GPU0 has already finished processing decode, which ends earlier.

### Takeaway 4

> Again, step sizes are NOT the same. This can lead to large variance in the compute time of LLM iterations, depending on the composition of prefill and decode tokens in the batch. This introduces significant bubbles when using pipeline parallelism.


## Sarathi-Serve

If we add up all the insights from the four takeaways (especially Takeaway 2), we can think of making hybrid batching "smarter" by carefully adding prefill computation. This will introduce little to no additional cost while increasing throughput.

To do this, Sarathi-Serve introduces chunked prefill and stall-free batching with a token budget.

### Chunked Prefill

Chunked prefill is a technique used in stall-free batching that allows "partial" prefill. Before this paper, prefill was processed atomically: either all or none.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-08-chunked-prefill-before.png"
    alt="Prefill processed atomically before chunked prefill"
    width="340"
  />
  <br />
  <sub>Figure 9. Before chunked prefill, a prefill request is handled atomically in one large step.</sub>
</p>

By using chunked prefill, we can process only part of it per step. This gives us flexibility in how much we process in one step. We will talk more about this in stall-free batching.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-09-chunked-prefill-after.png"
    alt="Chunked prefill split across multiple steps"
    width="340"
  />
  <br />
  <sub>Figure 10. Chunked prefill breaks a long prefill into smaller pieces that can be scheduled across steps.</sub>
</p>

### Overhead

Chunked prefill introduces additional overhead compared to standard prefill. In a regular prefill, all input tokens are processed in a single pass, so there is no need to revisit previously computed KV states.

With chunked prefill, however, the prefill phase is split across multiple iterations. Each new chunk must attend to tokens from earlier chunks, which requires repeatedly reading the previously computed KV cache during attention.

This results in extra memory reads due to KV cache re-access. 

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-10-chunked-prefill-overhead.png"
    alt="Chunked prefill overhead across chunk sizes"
    width="540"
  />
  <br />
  <sub>Figure 11. The overhead of chunked prefill stays modest and falls as chunk size grows.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

In practice, this overhead remains modest because prefill is largely compute-bound, so the additional KV reads have limited impact on overall runtime. According to the ablation study section, the overhead is at most around 25% for small chunk sizes and becomes negligible for larger chunks.


### Stall Free Batching

This is the scheduling algorithm introduced in Sarathi-Serve:

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-11-stall-free-batching-algorithm.png"
    alt="Sarathi-Serve stall-free batching algorithm"
    width="320"
  />
  <br />
  <sub>Figure 12. Sarathi-Serve's stall-free batching algorithm uses a fixed token budget for each step.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

If we break down the algorithm, stall-free batching works as follows:

1. **Initialize a token budget.**

   _Token budget_ is the fundamental knob used in Sarathi-Serve, which determines how many tokens to allow in a step. This is static throughout serving. We will talk more about the considerations and tradeoffs involved in setting the token budget.

2. **Also initialize batch number of tokens.**

   This is used to check whether the current batch has exceeded the token budget.

3. **For each step:**

   1. Prefer decode requests first.
   2. Update the batch number of tokens and the remaining token budget.
   3. Check whether there is a partially processed prefill request with tokens remaining.

      If yes:

      - Set the chunk size to `min(remaining budget, remaining prefill length)`.
      - If the remaining budget is `0`, end this step.
      - Otherwise, continue to the next sub-step.

   4. Update the batch number of tokens and the remaining token budget again.
   5. Get the next request and repeat the same logic until the token budget for this step is exhausted.

4. **Process hybrid batching of all the requests selected in step 3.**

To make this more concrete, I will walk through an illustrated example that hopefully makes the scheduling process more intuitive.

### Stall Free Batching w/ Illustrated Example

Suppose we have a token budget of 100.

With 4 requests R1~R4 in their decode phase, one chunked request (from an earlier step) R5(30 tokens), and two prefill requests R6(80 tokens) and R7(150 tokens):

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-13-illustrated-example-initial-state.png"
    alt="Initial state of the illustrated stall-free batching example"
    width="560"
  />
  <br />
  <sub>Figure 13. The initial state before scheduling begins under a token budget of 100.</sub>
</p>

`batch_num_tokens` is initialized to 0 for each step.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-14-illustrated-example-schedule-decodes.png"
    alt="First scheduling step admits decode requests"
    width="560"
  />
  <br />
  <sub>Figure 14. The scheduler first admits all decode requests in the current iteration.</sub>
</p>

First, we need to process all four decode requests, so we schedule them. `batch_num_tokens` becomes 4.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-15-illustrated-example-schedule-r5.png"
    alt="Scheduler adds the leftover chunked prefill request"
    width="560"
  />
  <br />
  <sub>Figure 15. The remaining chunked prefill request is added next within the unused token budget.</sub>
</p>

Since `batch_num_tokens` (4) did not reach the token budget (100), we continue.

We prioritize the chunked request, which is R5 in this case. `min(remaining_budget, R5) = 30`, so we schedule it to process the whole leftover prefill request. `batch_num_tokens` is updated to 34.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-16-illustrated-example-chunk-r6.png"
    alt="Scheduler chunks the next prefill request to fit the budget"
    width="560"
  />
  <br />
  <sub>Figure 16. The next prefill request is chunked so only the portion that fits the budget is scheduled.</sub>
</p>

The next request (in FCFS order) is R6, so we determine how much of it we can schedule. It turns out that `min(remaining_budget, R6) = 66`, so we chunk R6 and schedule only its first 66 tokens.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-17-illustrated-example-leftovers.png"
    alt="Leftover requests deferred to the next step"
    width="560"
  />
  <br />
  <sub>Figure 17. The unscheduled remainder is carried over to the next iteration.</sub>
</p>

The remainder stays pending for future iterations.

So R1 ~ (chunked) R6 are scheduled for this step and processed in hybrid batch.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-18-illustrated-example-final-batch.png"
    alt="Final hybrid batch chosen for the illustrated step"
    width="560"
  />
  <br />
  <sub>Figure 18. The final hybrid batch combines decode tokens with chunked prefill under one token budget.</sub>
</p>

For this iteration, the step latency can be approximated as $T_{\text{hybrid}} \approx T_{\text{batched non-attention}} + T_{\text{attention related operation for requests}} + T_{\text{overhead}}$

## How Sarathi-Serve solves the problem

Stall-free batching does use the hybrid batching technique introduced by Orca, but the difference is that instead of using the number of requests as an upper bound, it uses a token budget.

Before Sarathi-Serve, batched tokens per step had high variance because some steps batch together long prefills, while other steps consist mostly of decodes. Also, request input lengths differ, so some batched prefills take a very long time.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-19-token-budget-high-variance.png"
    alt="High per-step token variance before token-budgeted scheduling"
    width="480"
  />
  <br />
  <sub>Figure 19. Without a token budget, per-step work varies widely depending on the batch mix.<sup><a href="#reference-3">[3]</a></sup></sub>
</p>

By capping the token budget, we can control and estimate worst-case step latency much better, since it is bounded by a batch whose total tokens are approximately equal to the token budget. The token budget is usually capped to match the SLO requirement (e.g., < 10ms), so we can batch more prefill tokens while adding minimal latency to the decode steps batched with them.

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-20-token-budget-bounded-latency.png"
    alt="Bounded step latency with token-budgeted scheduling"
    width="600"
  />
  <br />
  <sub>Figure 20. A token budget bounds step latency and reduces variance across iterations.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

Also, since each step is capped to the token budget, variance becomes much lower, minimizing the pipeline bubbles.

### Considerations (Tradeoffs) for determining token budget

Choosing the token budget is a tradeoff among latency, efficiency, and hardware behavior.

1. TBT(TPOT) SLO (primary constraint)

   Since the point of having a token budget is to satisfy the decode SLOs, the budget must be small enough to avoid exceeding the requirement. If too many prefill tokens are included, decode iterations get delayed, causing generation stalls.

2. **Tile Quantization**  

   GPU kernels operate on fixed tile sizes (e.g., 16×16, 32×32). Misaligned chunk sizes can lead to wasted computation. For example, a chunk size of 257 can increase prefill time by ~32% due to poor tiling efficiency. (tbh I don't know how they manage to make this efficient because of my understanding, token budget only caps the upper bound)

3. **Pipeline Bubbles**  

   Larger chunks increase variation in iteration time, which leads to pipeline bubbles in pipeline-parallel setups. Smaller, more uniform chunks help maintain balanced execution across stages.

## Notes on evaluation

<p align="center">
  <img
    src="../assets/notes/sarathi-serve/sarathi-serve-21-evaluation-results.png"
    alt="Sarathi-Serve evaluation results under SLO-constrained serving"
    width="620"
  />
  <br />
  <sub>Figure 21. Under SLO-constrained serving, Sarathi-Serve improves serving capacity across models and setups.<sup><a href="#reference-1">[1]</a></sup></sub>
</p>

As a result of what we have talked about so far, Sarathi-Serve consistently improves serving capacity across models, with gains of about 2.6× to 5.6× depending on the model and parallelism, under realistic workloads.

Unlike benchmarks in some previous papers, Sarathi-Serve specifically measured throughput gains under SLOs, which I found interesting.

## References
<ol>
  <li id="reference-1">Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." <a href="https://arxiv.org/pdf/2403.02310">Link</a></li>
  <li id="reference-2">He, "Making Deep Learning Go Brrrr From First Principles." <a href="https://horace.io/brrr_intro.html">Link</a></li>
  <li id="reference-3">Agrawal, "Sarathi-Serve" OSDI '24 Slides. <a href="https://www.usenix.org/system/files/osdi24_slides-agrawal.pdf">Link</a></li>
  <li id="reference-4">BentoML, "LLM Inference Metrics." <a href="https://bentoml.com/llm/inference-optimization/llm-inference-metrics">Link</a></li>
  <li id="reference-5">Choi et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." <a href="https://dl.acm.org/doi/full/10.1145/3712285.3759823">Link</a></li>
  <li id="reference-6">Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." <a href="https://arxiv.org/pdf/2403.02310">Link</a></li>
  <li id="reference-7">Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models." <a href="https://www.usenix.org/system/files/osdi22-yu.pdf">Link</a></li>
  <li id="reference-8">UvA DL Notebooks, "Pipeline Parallelism." <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/pipeline_parallel_simple.html">Link</a></li>
  <li id="reference-9">Weng, "How to Train Really Large Models on Many GPUs." <a href="https://lilianweng.github.io/posts/2021-09-25-train-large/#pipeline-parallelism">Link</a></li>
  <li id="reference-10">Rao, "Distributed and Efficient Finetuning." <a href="https://sumanthrh.com/post/distributed-and-efficient-finetuning/">Link</a></li>
  <li id="reference-11">"Improving LLM Inference with Continuous Batching: Orca through tinyorca." <a href="./tinyorca.md">Link</a></li>
  <li id="reference-12">"Introduction to LLM Inference Part 1." <a href="./llm-inference-intro-p1.md">Link</a></li>
</ol>
