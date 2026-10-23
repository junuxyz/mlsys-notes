"""Profile one decode forward pass of Standard and merged absorption on H100.

Run: uv run --project labs modal run labs/mla-lab/profile.py
Saves kernel timings and Chrome traces (open in ui.perfetto.dev) under results/profile/.
"""

import json
import runpy
import sysconfig
import tempfile
from pathlib import Path

_stdlib_profile = runpy.run_path(str(Path(sysconfig.get_path("stdlib")) / "profile.py"))
_Utils = _stdlib_profile["_Utils"]
run = _stdlib_profile["run"]
runctx = _stdlib_profile["runctx"]

import measure  # noqa: E402
import modal  # noqa: E402

app = modal.App("mla-lab-profile")


@app.function(
    image=measure.image.add_local_file(
        Path(__file__).with_name("measure.py"), "/root/measure.py"
    ),
    gpu="H100!",
    cpu=4,
    timeout=1200,
)
def profile_decode():
    import flashinfer
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    data = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "cuda": torch.version.cuda,
        "q_len": 1,
        "kv_len": measure.KV_LEN,
        "scope": "one warmed forward pass per method, profiled separately",
        "profiles": {},
    }
    traces = {}
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
        operations = {
            # Full forward passes (h to output), one profiler session each.
            "standard_forward": measure.make_case(1, measure.KV_LEN, "standard")[0],
            "merged_forward": measure.make_case(1, measure.KV_LEN, "merged_absorb")[0],
        }
        for name, operation in operations.items():
            for _ in range(10):
                _output = operation()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                _output = operation()
                torch.cuda.synchronize()

            print(f"\n{name}", flush=True)
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
            )
            operators = [
                {
                    "name": event.key,
                    "count": event.count,
                    "gpu_us": event.self_device_time_total,
                    "shapes": event.input_shapes,
                }
                for event in prof.key_averages(group_by_input_shape=True)
                if event.device_type == torch.autograd.DeviceType.CPU
                and event.self_device_time_total > 0
            ]
            kernels = [
                {
                    "name": event.name,
                    "start_us": event.time_range.start,
                    "gpu_us": event.time_range.elapsed_us(),
                }
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
                and not event.name.startswith(("Memcpy", "Memset"))
            ]
            first = min((k["start_us"] for k in kernels), default=0)
            for kernel in kernels:
                kernel["start_us"] -= first
            data["profiles"][name] = {
                "operators": operators,
                "kernels": kernels,
            }
            trace = Path(temp) / f"{name}.json"
            prof.export_chrome_trace(str(trace))
            traces[name] = trace.read_text()
    return data, traces


@app.local_entrypoint()
def main():
    data, traces = profile_decode.remote()
    # Overwrites the previous run.
    destination = Path(__file__).with_name("results") / "profile"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    for method, trace in traces.items():
        (destination / f"{method}.trace.json").write_text(trace)
    print(f"Saved profiles: {destination}")
