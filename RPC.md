# RPC pool: building the GGML_RPC image

Llama Launcher's "RPC pool" mode drives `llama.cpp`'s built-in `rpc-server` /
`--rpc` mechanism to pool VRAM (and, worker-side, RAM) across several
processes and machines. This is a **capacity** feature, not a speed feature:
splitting a model across an RPC pool is slower than running it on one card,
but it lets a model that doesn't fit on any single node's VRAM load at all.

## Why you have to build your own image

The prebuilt `ghcr.io/ggml-org/llama.cpp` images are **not** built with
`GGML_RPC=ON`, so they contain neither the `--rpc` flag on `llama-server` nor
the `rpc-server` binary. There is no supported prebuilt RPC image as of this
writing (spike-verified 2026-08-20) — you must build one yourself.

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

`GGML_RPC=ON` is the flag that matters — it compiles both the `--rpc` client
support into `llama-server` and the standalone `rpc-server` binary. If the
project's `cuda.Dockerfile` doesn't already thread `GGML_RPC` through to
`cmake`, add `-DGGML_RPC=ON` to its `cmake` invocation directly and rebuild.

### Verify the build

```bash
# llama-server must advertise --rpc:
podman run --rm --entrypoint /app/llama-server llama.cpp:rpc-cuda --help | grep -i rpc
#   --rpc SERVERS     ...

# rpc-server must exist in the image:
podman run --rm --entrypoint /bin/sh llama.cpp:rpc-cuda -c 'ls -l /app/rpc-server'
```

If either check fails, the build didn't actually turn RPC on — re-check the
`GGML_RPC` plumbing before trying to launch a pool against this tag.

## Local dry-run (one box, two simulated workers)

This exercises the same wiring the GUI uses for a real multi-node pool,
just with every worker on localhost.

1. Start two `rpc-server` workers, each in its own container, published to
   loopback only (workers are never meant to be reachable off-box except
   through an SSH tunnel — see below):

   ```bash
   podman run -d --name rpc-w0 --device nvidia.com/gpu=all \
     -p 127.0.0.1:50052:50052 --entrypoint /app/rpc-server \
     llama.cpp:rpc-cuda -H 0.0.0.0 -p 50052 -d CUDA0

   podman run -d --name rpc-w1 \
     -p 127.0.0.1:50053:50053 --entrypoint /app/rpc-server \
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
  on its own host — never to a routable interface.
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

## One build, every node

`rpc-server` and `llama-server` talk to each other over ggml's own wire
format, which is **not** version-stabilized across builds. Every node in a
pool — the head and all workers, local or remote — must run the **same**
`llama.cpp` build (i.e. the same image tag, rebuilt and re-pushed/re-pulled
together whenever you update). Mixing builds across nodes is the most common
cause of an RPC pool that connects but fails or misbehaves mid-load.
