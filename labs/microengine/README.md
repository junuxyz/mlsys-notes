## microengine

<p align="center">
  <img src="../../assets/labs/microengine/microengine_1.png" alt="microengine diagram" width="480" />
</p>

`microengine` is a minimal serving system built to keep the core serving components small, readable, and centered on
  the essential moving parts:

- an endpoint
- requests
- a FIFO request queue
- a static batch scheduler
- an execution engine
- a model runner


## What it includes

- FIFO request admission
- static batching with `max_batch_size`
- one prefill pass per admitted batch
- iterative decode until each request finishes
- simple stopping rules: EOS or `max_new_tokens`
- a tiny demo in [example.py](./example.py)

## Mental model

The serving system is easiest to read as this loop:

1. Submit requests into the queue.
2. The endpoint tokenizes them and admits them into the queue.
3. The scheduler picks the next static batch.
4. The engine runs prefill once for that batch.
5. The engine repeatedly decodes one token step at a time.
6. Move on to the next queued batch.

That's it.


> Note: This is intentionally a batch-to-completion, static-batching serving system.
Real serving systems usually add more machinery on top of this core loop, such as token streaming, scheduler re-entry,
request-level metrics, and more flexible execution policies.


## Components and responsibilities

Even across modern inference frameworks, component names and responsibilities differ a bit. There is no single source of truth for how a serving system should be decomposed.

Because of that, `microengine` is best treated as a small mental model for understanding the serving loop holistically rather than a canonical architecture.

With that framing, `microengine` keeps the serving loop split into a few explicit pieces:

- `RequestState`
  
  Represents the lifecycle of one request: `WAITING`, `RUNNING`, or `FINISHED`.

- `Request`
  
  Holds the per-request state.
  This is the object that carries the prompt tokens, generated tokens, stopping configuration, and lifecycle state.

- `RequestQueue`
  
  Owns admission order.
  It is just a FIFO queue of waiting requests.

- `Endpoint`

  Owns ingress.
  It tokenizes raw prompt text, builds `Request` objects, and admits them into the queue.

- `StaticBatchScheduler`
  
  Decides which requests enter the next batch.
  In this engine, the policy is intentionally simple: take the next `max_batch_size` requests from the queue and run that batch to completion.

- `BatchState`
  Holds the request states for one active batch.
  It keeps the active requests, the next input tokens, the attention mask, and the KV cache.

- `ModelRunner`
  Owns the model forward passes.
  It does two things:
  1. `prefill()` runs the prompt pass for a newly admitted batch.
  2. `decode()` advances that batch by one token.

- `MicroEngine`

  Owns the data-plane execution for one selected batch.
  It calls the model runner, updates each request, and drives the decode loop until the batch finishes.

- `ServingSystem`

  Wires the subsystems together.
  It owns the endpoint, request queue, scheduler, and engine, and orchestrates the overall serving loop.

## Example

Run the demo directly:

```bash
uv run python -m labs.microengine.example
```

The example constructs a `ServingSystem`, submits a few prompts, runs the system, and prints each final decoded output.


## Todo
- [ ] implement memory allocation without using pytorch
- [ ] TinyLlama -> Qwen
