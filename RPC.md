# RPC pool: building the GGML_RPC image

Llama Launcher's "RPC pool" mode drives `llama.cpp`'s built-in `rpc-server` /
`--rpc` mechanism to pool VRAM (and, worker-side, RAM) across several
processes and machines. This is a **capacity** feature, not a speed feature:
splitting a model across an RPC pool is slower than running it on one card,
but it lets a model that doesn't fit on any single node's VRAM load at all.

## Why you have to build your own image

The prebuilt `ghcr.io/ggml-org/llama.cpp` images are **not** built with
`GGML_RPC=ON`, so they contain neither the `--rpc` flag on `llama-server` nor
the `rpc-server` binary. There is no supported prebuilt RPC image; you must
build one yourself.

One image serves double duty: it runs `llama-server` (the pool **head**) on
one node and `rpc-server` (a pool **worker**) on every other node. Build it
once, tag it, and reuse the same tag everywhere in the pool.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

podman build -t llama.cpp:rpc-cuda \
  -f .devops/cuda.Dockerfile \
  --build-arg CUDA_DOCKER_ARCH=all \
  --build-arg GGML_RPC=ON \
  .
```

`GGML_RPC=ON` is the flag that matters: it compiles both the `--rpc` client
support into `llama-server` and the standalone `rpc-server` binary. If the
project's `cuda.Dockerfile` doesn't already thread `GGML_RPC` through to
`cmake`, add `-DGGML_RPC=ON` to its `cmake` invocation directly and rebuild.

### Verify the build

```bash
# llama-server must advertise --rpc:
podman run --rm --entrypoint /app/llama-server llama.cpp:rpc-cuda --help | grep -i rpc
#   --rpc SERVERS     ...

# the RPC worker binary must exist. Current llama.cpp names it
# `ggml-rpc-server` (older builds called it `rpc-server`):
podman run --rm --entrypoint /bin/sh llama.cpp:rpc-cuda -c 'ls -l /app/ggml-rpc-server'
```

The worker binary's only options are `-t/-d/-H/-p/-c`; there is **no**
per-worker `--mem` budget flag in current builds, so Llama Launcher treats a
worker's memory pledge as a "Check fit" preflight number only and never passes
it to the worker.

If either check fails, the build didn't actually turn RPC on; re-check the
`GGML_RPC` plumbing before trying to launch a pool against this tag.

## Local dry-run (one box, two simulated workers)

This exercises the same wiring the GUI uses for a real multi-node pool,
just with every worker on localhost.

1. Start two `rpc-server` workers, each in its own container, published to
   loopback only (workers are never meant to be reachable off-box except
   through an SSH tunnel; see below):

   ```bash
   podman run -d --name rpc-w0 --device nvidia.com/gpu=all \
     -p 127.0.0.1:50052:50052 --entrypoint /app/ggml-rpc-server \
     llama.cpp:rpc-cuda -H 0.0.0.0 -p 50052 -d CUDA0

   podman run -d --name rpc-w1 \
     -p 127.0.0.1:50053:50053 --entrypoint /app/ggml-rpc-server \
     llama.cpp:rpc-cuda -H 0.0.0.0 -p 50053 -d CPU
   ```

2. Start the head, host-networked so its loopback can see both workers'
   published ports, pointing `--rpc` at both:

   ```bash
   podman run -d --name rpc-head --network host \
     --device nvidia.com/gpu=all --entrypoint /app/llama-server \
     -v /path/to/models:/models:ro \
     llama.cpp:rpc-cuda -m /models/your-model.gguf \
     --rpc 127.0.0.1:50052,127.0.0.1:50053 \
     --host 0.0.0.0 --port 8080
   ```

   Watch the head's logs: the model's tensors should load spread across the
   local CUDA device and the (simulated) CPU worker.

3. Tear down: stop the head first, then both workers
   (`podman stop rpc-head rpc-w0 rpc-w1`). Confirm nothing lingers with
   `podman ps -a | grep llama-`.

## How the GUI automates this

An "RPC pool" profile in Llama Launcher does exactly the above, generalized
to real remote nodes:

- Every worker container publishes its `rpc-server` port to `127.0.0.1` only
  on its own host, never to a routable interface.
- A **local** worker (same node as the GUI/head) is reached directly via
  `127.0.0.1:<port>`.
- A **remote** worker is reached through an `ssh -L <port>:127.0.0.1:<port>`
  tunnel opened to that node, so from the head's point of view every worker
  is still just a loopback address.
- The head container runs with `--network host` and `--rpc
  127.0.0.1:<port1>,127.0.0.1:<port2>,...` listing every worker (local and
  tunnelled) by its loopback port.
- Launch order is workers-first: the launcher waits for every worker's port
  to be listening before starting the head. Stop order is head-first, then
  workers, then any open tunnels are closed.
- Launch-time validation does not ssh-probe every node, so use the GUI's
  "Check fit" button before launching a pool: it pre-checks per-node
  RPC-image presence and VRAM/RAM fit up front, instead of failing partway
  through a live launch.

## One build, every node

`rpc-server` and `llama-server` talk to each other over ggml's own wire
format, which is **not** version-stabilized across builds. Every node in a
pool (the head and all workers, local or remote) must run the **same**
`llama.cpp` build (i.e. the same image tag, rebuilt and re-pushed/re-pulled
together whenever you update). Mixing builds across nodes is the most common
cause of an RPC pool that connects but fails or misbehaves mid-load.

## Verifying a real multi-node pooled run: what to look for

A pool can "start" (workers up, head running) and still not actually be
pooling, or fail subtly mid-load. Use these signals to tell the difference.

> **Under construction.** RPC pooling is implemented but has not yet been
> verified by the author on real multi-GPU hardware, and the build is still
> being planned out. Treat the whole feature as experimental and expect rough
> edges; feedback and issue reports are welcome.

### Success signals: the head log at load time

- The RPC devices appear in the head's device list, e.g. `RPC[<host>:<port>]`
  alongside your local `CUDA0`.
- The `load_tensors:` summary prints a **non-zero buffer size per RPC device**
  (`RPC[…] model buffer size = … MiB`). This line is the authoritative record
  of where each layer landed. If an RPC device shows ~0, no layers went to
  that worker.
- The model finishes loading and `/health` returns `200` **without**
  `Remote RPC server crashed or returned malformed response`.
- Each worker log shows `Accepted client connection` plus allocations, no crash.

### Confirm it is actually distributed (not all on the head)

- Run `nvidia-smi` **on each worker node** during load/inference: the
  `ggml-rpc-server` process should be holding VRAM. A worker sitting at ~0 MiB
  got no layers.
- **`-ngl` (GPU-layers) must be high** (e.g. `99`) to offload layers onto the
  RPC devices at all; without it, layers stay on the head and workers idle.
- Use **`--tensor-split`** on the head if the automatic split overcommits one
  node.

### Failure modes and how to recognize them

1. **Wire-format mismatch (most common):** head and a worker on *different*
   llama.cpp builds → connects, then malformed-response/garbage mid-load. Fix:
   the **same image tag on every node** (see "One build, every node").
2. **Worker OOM:** there is no per-worker `--mem` cap, so a worker exposes its
   full VRAM; if the split over-allocates, it OOM-kills mid-upload and the head
   reports "server crashed." Watch each node's VRAM headroom; that is what the
   GUI's "Check fit" estimates.
3. **`[create_node] invalid data ptr` / graph-compute crash:** observed with
   **CPU-device** workers (`-d CPU`) on current llama.cpp; the CPU-donation
   path appears buggy upstream. GPU workers are the supported path; if you see
   this with GPU workers, suspect a version mismatch (#1).
4. **Dead ssh tunnel / unreachable worker:** the launcher's readiness gate
   refuses to start the head ("worker N failed to start"). Check that node's
   image tag and that its tunnel came up.
5. The rpc-server's "**Never expose the RPC server to an open network**"
   warning is **expected**: loopback publish + ssh tunnels keep it private.

### Performance expectation

Pooling is **slower than single-box**: it is a capacity feature, not a speed
feature; each token serializes tensors over the link. Catastrophic slowness
usually means the interconnect (LAN bandwidth) is the bottleneck; RPC is
bandwidth-sensitive. Avoid `--cpu-moe` / `--n-cpu-ffn` / `--no-kv-offload`
(they centralize work on the head; the launcher warns about them).

### In the GUI

- Run **"Check fit"** first and sanity-check its pooled VRAM+RAM headline
  against real per-node `nvidia-smi` free memory.
- Known cosmetic quirk: a **worker card's tok/s reflects the head's** HTTP
  metrics (an `rpc-server` has no HTTP endpoint). Judge a worker's health by
  its container being up and its VRAM in `nvidia-smi`, not its card number.
