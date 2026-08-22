# Recovering NVLS (NVLink SHARP) on H100 nodes

This documents the recovery procedure used when NCCL could see NVLink SHARP (NVLS), but could not allocate its multicast resources on either 8-GPU H100 NVSwitch node.

## Problem

When trying to set up DeepSeekV4-Flash with TP=4, the worker failed while initializing its NCCL communicator. We inspected the previous container log (edited the following trace for better readability):

```bash
kubectl -n dsv4-bench logs "$POD" \
  --previous \
  --timestamps \
  | grep -Ei -B 20 -A 40 \
  'NCCL (WARN|INFO)|Cuda failure|unhandled cuda|peer access|cuMem|NVLink|shm|topology' \
  | tail -n 300
```

Output:

```bash
dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1

dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] transport/nvls.cc:379 (nvlsAllocateMem) NCCL WARN Failed to bind NVLink SHARP (NVLS) Multicast memory of size 2097152 : CUDA error 802 'system not yet initialized'.
This is usually caused by a system or configuration error in the Fabric Manager or NVSwitches.
Disable NVLS (NCCL_NVLS_ENABLE=0) if you wish to avoid this error in the future.

dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO transport/nvls.cc:558 (ncclNvlsSetup) -> 1
dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO init.cc:1558 (initTransportsRank) -> 1
dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO init.cc:1921 (ncclCommInitRankFunc) -> 1
dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO init.cc:2545 (ncclCommInitRankDev) -> 1
dsv4-flash-baseline-0-vllmworker-3:2736:2736 [0] NCCL INFO init.cc:2572 (ncclCommInitRank) -> 1

(Worker pid=2730) ERROR [multiproc_executor.py:912] WorkerProc failed to start.
(Worker pid=2730) ERROR [gpu_worker.py:376] init_worker_distributed_environment(
(Worker pid=2730) ERROR [parallel_state.py:2009] ensure_model_parallel_initialized(
(Worker pid=2730) ERROR [cuda_communicator.py:86] self.pynccl_comm = PyNcclCommunicator(
(Worker pid=2730) ERROR [pynccl.py:137] self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
(Worker pid=2730) ERROR [pynccl_wrapper.py:417] raise RuntimeError(f"NCCL error: {error_str}")
(Worker pid=2730) ERROR RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)
```

Based on the log, we could tell NCCL failed specifically while binding [NVLink SHARP (NVLS)](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/env.html#nccl-nvls-enable)
multicast memory in `nvlsAllocateMem`.

## Solution (TL;DR)

The failure was caused by an out-of-sync NVLink fabric state. Even after rebooting, although Fabric Manager was running and all GPUs reported Completed/Success, the kernel reported `NV_ERR_FABRIC_STATE_OUT_OF_SYNC` when NCCL attempted to allocate NVLS multicast memory.

We recovered by stopping all GPU clients, resetting all GPUs and NVSwitches with `nvidia-smi -r` (when no GPU is specified, the command resets all GPUs on the node), and then restarting Fabric Manager.

So the tip is: before resetting the GPUs, make sure to stop all CUDA workloads and GPU-related management services that hold NVIDIA devices.

This includes:
- Fabric Manager
- DCGM
- nvidia-snapshot
- and GPU Operator components (e.g. device plugin, MIG manager, and GPU Feature Discovery)

Otherwise, `nvidia-smi -r` may refuse the reset because NVIDIA devices are still in use.

## Background

So, what is NVLS? NVLS (NVLink SHARP) allows the NVSwitch fabric in H100 systems to accelerate collective operations such as all-reduce and all-gather.

We did not encounter this issue during previous Qwen3-32B-FP8 TP=2 experiments. We left `NCCL_NVLS_ENABLE` unset, allowing NCCL to select whether to use NVLS automatically. NCCL may not have selected NVLS for that workload. In the DeepSeekV4-Flash configuration, however, NCCL selected NVLS and failed during initialization.

While we could simply add `- {name: NCCL_NVLS_ENABLE, value: "0"}` to the `deploy.yaml` file, this meant we don't use NVLS within the pods, and use a NCCL fallback. This was a workaround that could reduce performance when NCCL would otherwise select the accelerated NVLS collective path. So instead, we tried to fix what went wrong.

## Recovery attempts

### Inspect Fabric Manager

Log message already clearly showed what we should look into next:

```
mem_multicast_fabric.c:3039
NV_ERR_FABRIC_MANAGER_NOT_PRESENT
Fabric Manager is not loaded
```

Fabric manager was "NOT_PRESENT". So we decided to see the [Fabric Manager(FM)](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html#what-is-fabric-manager) state in the node:

```bash
echo "=== DRIVER ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u

echo "=== FABRIC REGISTRATION ==="
nvidia-smi -q | grep -i -A 2 Fabric

echo "=== FABRIC MANAGER VERSION ==="
nv-fabricmanager --version 2>&1 || true

echo "=== FABRIC MANAGER SERVICE ==="
sudo systemctl status nvidia-fabricmanager --no-pager -l || true

echo "=== RECENT FABRIC MANAGER LOG ==="
sudo journalctl -u nvidia-fabricmanager -b --no-pager -n 100
```

This printed out:

```bash
=== DRIVER ===
580.xxx.xx

=== FABRIC REGISTRATION ===
        GPU Fabric GUID                   : N/A
    Inforom Version
        Image Version                     : G520.0200.00.05
--
    Fabric
        State                             : Completed
        Status                            : Success
--
        GPU Fabric GUID                   : N/A
    Inforom Version
        Image Version                     : G520.0200.00.05
--
    Fabric
        State                             : Completed
        Status                            : Success
--
...
--
    Fabric
        State                             : Completed
        Status                            : Success
=== FABRIC MANAGER VERSION ===
Fabric Manager version is : 580.xxx.xx
=== FABRIC MANAGER SERVICE ===
× nvidia-fabricmanager.service - NVIDIA fabric manager service
     Loaded: loaded (/lib/systemd/system/nvidia-fabricmanager.service; enabled; vendor preset: enabled)
     Active: failed (Result: signal) since Mon 2026-08-10 15:45:34 UTC; 5 days ago
   Main PID: 9946 (code=killed, signal=KILL)
        CPU: 27.747s

Aug 10 06:41:31 inst-1onle-devrel-rdma-pool systemd[1]:
...
```

We figured out that in gpu05, `nvidia-fabricmanager.service` had failed after its main process was terminated with a `SIGKILL`. This happened quite a long ago (even before I participated in this project) so I didn't have the context of this. Anyways, we needed to recover the Fabric Manager in order to make NVLS work.

We decided to use `nvidia-smi -r` which resets the GPU set.

```bash
sudo systemctl stop nvidia-fabricmanager
sudo nvidia-smi -r
sudo systemctl reset-failed nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
```

This didn't work first and hit an error message:

```
8 devices are currently being used by one or more other processes (e.g., Fabric Manager, CUDA application, graphics application such as an X server, or a monitoring application such as another instance of nvidia-smi). Please first kill all processes using these devices and all compute applications running in the system.
```

The thing was, even if we don't have any CUDA process running in `nvidia-smi`, continuously running monitoring processes such as DCGM, `nvidia-snapshot`, or the device plugin can block the reset process.

#### Stop monitoring processes

So we needed to stop nvidia-snapshot that run once per minute:

```
sudo systemctl stop nvidia-snapshot.timer
sudo systemctl stop nvidia-snapshot.service 2>/dev/null || true
```

A parent `watch` process kept launching new `nvidia-smi` processes, which produced multiple PIDs. We terminated both the watcher and its child processes:

```
pkill -u "$USER" -f '[w]atch.*nvidia-smi' 2>/dev/null || true
pkill -u "$USER" -x nvidia-smi 2>/dev/null || true

sudo systemctl stop nvidia-fabricmanager
sleep 3
```

The reset was still blocked because NVIDIA devices remained in use.

### First Attempt: Reboot and test NVLS

One option was to use `sudo reboot` to reboot the node and stop all processes. While we've tried this, this alone didn't solve the problem! (Also if you do need to reboot for some other reason, recommend using the Restart option using the console manager if you're using Cloud environment, instead of directly rebooting from CLI)

```bash
nvidia-smi -q |
  grep -i -A 2 Fabric

sudo dmesg -T |
  grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' |
  tail
active
        GPU Fabric GUID                   : N/A
    Inforom Version
        Image Version                     : G520.0200.00.05
--
# ...
--
        GPU Fabric GUID                   : N/A
    Inforom Version
        Image Version                     : G520.0200.00.05
--
    Fabric
        State                             : Completed
        Status                            : Success
```

FM was now enabled. To validate whether the reboot had fixed NVLS, I ran a quick smoke test using `all_reduce_perf` from [nccl-tests](https://github.com/nvidia/nccl-tests):

```bash
export EXP_DIR=/ephemeral/shared/nvls-recovery
mkdir -p "$EXP_DIR"

set -o pipefail

NCCL_NVLS_ENABLE=1 \
NCCL_DEBUG=INFO \
/usr/bin/all_reduce_perf \
  -b 8M \
  -e 128M \
  -f 2 \
  -g 8 \
  2>&1 | tee "$EXP_DIR/nvls-smoke-before-reset-$(hostname -s).log"

echo "exit code: $?"
```

Even after Fabric Manager was enabled, NVLS multicast still made an error:

```
Bootstrap timings total 0.117246 (create 0.000016, send 0.000134, recv 0.020848, ring 0.095836, delay 0.000000)
xxx-xxx-devrel-rdma-pool:73980:74441 [4] NCCL INFO MNNVL busId 0x89000 fabric UUID 0.0 cliqueId 0x0 state 3 healthMask 0x0
...
xxx-xxx-devrel-rdma-pool pid 73980: Test failure common.cu:893
exit code: 3
```

We checked for other reasons and found a log:

```bash
sudo journalctl -k -b --no-pager |
grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' |
tail -n 2

xxx-xxx-devrel-rdma-pool kernel: NVRM: nvCheckOkFailedNoLog: Check failed: NVLink fabric state cached by the driver is out of sync
...
```

So the current state was: `nvidia-fabricmanager` is active and all eight GPUs report Fabric Completed/Success, but the kernel reports `NV_ERR_FABRIC_STATE_OUT_OF_SYNC` when NVLS multicast is attempted.

### Second attempt: Proper `nvidia-smi -r`

During the first attempt, we noticed `nvidia-smi -r` never completed properly and we just forcefully rebooted. This time, we tried stopping the remaining GPU users before retrying the reset.

#### Stop all GPU clients

```bash
sudo systemctl stop nvidia-snapshot.timer 2>/dev/null || true
sudo systemctl stop nvidia-snapshot.service 2>/dev/null || true
sudo systemctl stop kubelet
sudo systemctl stop nvidia-fabricmanager 2>/dev/null || true

sleep 5

echo "=== SERVICE STATE ==="
systemctl is-active kubelet || true
systemctl is-active nvidia-fabricmanager || true

echo "=== GPU MANAGEMENT PROCESSES ==="
pgrep -af \
  'dcgm-exporter|nv-hostengine|nvidia-device-plugin|nvidia-mig-manager|gpu-feature-discovery' \
  || true

echo "=== OPEN NVIDIA DEVICES ==="
sudo fuser -v \
  /dev/nvidiactl \
  /dev/nvidia-uvm \
  /dev/nvidia-uvm-tools \
  /dev/nvidia-modeset \
  /dev/nvidia-nvlink \
  /dev/nvidia-nvswitchctl \
  /dev/nvidia-nvswitch{0..3} \
  /dev/nvidia{0..7} 2>&1
```

This time we also stopped `kubelet` which had remained running during the first attempt. Stopping kubelet prevented Kubernetes from recreating the GPU Operator pods.

However, it did not necessarily terminate containers that were already running. Our first process check still showed:

```bash
37464 nvidia-mig-manager
37538 /usr/bin/dcgm-exporter
38665 gpu-feature-discovery
38668 nvidia-device-plugin
924303 /snap/dcgm/62/usr/bin/nv-hostengine -n
```

We verified that each PID belonged to GPU management processes and sent `SIGTERM` only to those verified PIDs. After that, we checked again.

Output after stopping them:

```bash
=== SERVICE STATE ===
inactive
failed
=== GPU MANAGEMENT PROCESSES ===
=== OPEN NVIDIA DEVICES ===
```

This time, finally we were able to stop all processes preventing the GPU reset.

#### Reset the GPUs and NVSwitches

```bash
echo "RESET TARGET: $(hostname -s)"
REPAIR_STARTED=$(date '+%Y-%m-%d %H:%M:%S')

sudo nvidia-smi -r
RESET_RC=$?

echo "GPU reset exit code: $RESET_RC"

if [ "$RESET_RC" -ne 0 ]; then
  echo "Reset failed; restoring services"

  sudo systemctl reset-failed nvidia-fabricmanager
  sudo systemctl start nvidia-fabricmanager || true
  sudo systemctl start kubelet
  sudo systemctl start nvidia-snapshot.timer 2>/dev/null || true

  systemctl is-active nvidia-fabricmanager || true
  systemctl is-active kubelet || true
else
  echo "Reset succeeded; starting Fabric Manager"

  sudo systemctl reset-failed nvidia-fabricmanager
  sudo systemctl start nvidia-fabricmanager

  sleep 10

  echo "=== FABRIC MANAGER ==="
  systemctl is-active nvidia-fabricmanager || true

  echo "=== FABRIC REGISTRATION ==="
  nvidia-smi -q |
    grep -i -A 2 '^    Fabric$'

  echo "=== NEW FABRIC ERRORS ==="
  sudo journalctl -k \
    --since "$REPAIR_STARTED" \
    --no-pager |
    grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' || true

  echo "kubelet remains stopped intentionally"
fi
```

output:

```
GPU 00000000:0F:00.0 was successfully reset.
...
GPU 00000000:D8:00.0 was successfully reset.
Note: The operation has successfully reset all GPUs and NVSwitches. If the services, such as nvidia-fabricmanager, which manage or monitor NVSwitches are running, they might have been affected by this operation. Please refer respective service status or logs for details.
All done.
GPU reset exit code: 0
Reset succeeded; starting Fabric Manager
=== FABRIC MANAGER ===
active
=== FABRIC REGISTRATION ===
    Fabric
        State                             : Completed
        Status                            : Success
...
=== NEW FABRIC ERRORS ===
kubelet remains stopped intentionally
```

Finally we saw a sign of fix. Just to make sure we ran the same smoke test:

#### Validate NVLS

```bash
export EXP_DIR="${EXP_DIR:-/ephemeral/shared/nvls-recovery}"
mkdir -p "$EXP_DIR"

SMOKE_STARTED=$(date '+%Y-%m-%d %H:%M:%S')
SMOKE_LOG="$EXP_DIR/nvls-smoke-after-reset-$(hostname -s).log"

set -o pipefail

NCCL_NVLS_ENABLE=1 \
NCCL_ALGO=NVLS \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH \
/usr/bin/all_reduce_perf \
  -b 8M \
  -e 128M \
  -f 2 \
  -g 8 \
  2>&1 | tee "$SMOKE_LOG"

SMOKE_RC=$?
echo "NVLS smoke exit code: $SMOKE_RC"

grep -E \
  'NVLS multicast support|NCCL_ALGO set|nvls channels|Out of bounds|Avg bus bandwidth|NCCL WARN|Test failure|Cuda failure|unhandled cuda' \
  "$SMOKE_LOG" | tail -n 100

sudo journalctl -k --since "$SMOKE_STARTED" --no-pager |
  grep -E 'FABRIC_STATE_OUT_OF_SYNC|FABRIC_MANAGER_NOT_PRESENT' || true
```

output:

```bash
...
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 291.942
#

NVLS smoke exit code: 0
```

This time, it **worked** properly without any error. Since NVLS didn't work properly on either node, we went through the same process on the other node.

### Restore services

After the smoke test, we restored the services that had been stopped:

```bash
sudo systemctl start kubelet
sudo systemctl start nvidia-snapshot.timer 2>/dev/null || true

systemctl is-active nvidia-fabricmanager
systemctl is-active kubelet
systemctl is-active nvidia-snapshot.timer 2>/dev/null || true
```


Why did rebooting alone failed to fix the problem?

The first attempt did not successfully reset all GPU and NVSwitch state. We restored the clean state only after stopping the remaining GPU clients, running `nvidia-smi -r`, and restarting Fabric Manager in that order, as described [here](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#initializing-nvswitch-and-nvlink).


## References

- [NVIDIA Fabric Manager user guide](https://docs.nvidia.com/hgx-platforms/fabric-manager-user-guide/index.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
