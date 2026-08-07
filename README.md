# Llama Launcher

A PySide6/Qt6 desktop app that builds a `podman`/`docker` `llama-server` command from a GUI
and launches it in a terminal (konsole, foreground so `Ctrl-C` works), then observes the named
container from outside. Tabbed **Configure / Monitor** UI with profiles, a curated `llama-server`
settings catalog (plus a raw-args escape hatch), typed mounts, mmproj/LoRA/draft-model pickers,
model-aware capability detection, VRAM preflight, a live throughput/MTP monitor, and a
repeatable speed benchmark for A/B-ing config changes.

## Requirements

- Python ≥ 3.13
- `podman` (or `docker`) on `PATH`
- An NVIDIA GPU with the driver + [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) installed
- A terminal emulator (konsole by default)

## Install & run

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/llama-launcher                                   # GUI
# or: .venv/bin/python -m llama_launcher.app

.venv/bin/python -m llama_launcher.app --dry-run --profile "NAME"   # print the command, don't launch
.venv/bin/pytest -q                                        # test suite (offscreen Qt)
```

## GPU passthrough

The launcher offers two GPU modes (Configure tab → **Runtime**):

- **CDI — `--device nvidia.com/gpu=all`** (recommended) — uses the NVIDIA Container Device
  Interface spec.
- **Legacy — `--gpus all`** — the older runtime hook; doesn't read the CDI spec.

## Benchmark (Monitor tab)

The Monitor tab has a **Benchmark** section for a controlled, repeatable speed
measurement so a config change (flags or model) can be compared apples-to-apples.
It POSTs standardized filler prompts to the running server and reads llama.cpp's
`timings` to report **prompt-eval** and **generation** tok/s at each prompt size.

Configure the run inline: prompt **sizes** (default `128, 512, 2048` tokens),
**n_predict** (128), **warmup** runs (1, discarded), and **repeats** (3, averaged).
Requests always use `temperature 0` / `stream false`. **Run** is enabled only when
the server is running and ready (in router mode, also only while a model is
loaded); results fill a table of size · prompt_n · pp t/s · gen t/s · total s.

Each run is saved to a **per-profile history** (last 5, newest first), labelled
with a compact snapshot of the config it ran under (e.g. `-ngl99 fa=on`). The
latest run shows a **delta** versus the previous one (e.g. `Δ pp +8% · gen +3%`),
so changing a flag and re-running tells you immediately whether it helped. Works
for both single-model **server** and **router** profiles (router scopes the
request to the loaded model).

## Headless control

Drive a profile without the GUI (for test harnesses). Runs on the host with
podman + the GPU; the harness connects in over the network.

    llama-launcher --launch --profile PROFILE [--wait[=SECONDS]]
    llama-launcher --stop   --profile PROFILE
    llama-launcher --health --profile PROFILE

`--profile` falls back to the last-used profile. `--launch --wait` blocks until
the server answers `/health` (default 60s; models still load on demand).

Works for **router** and single-model **server** profiles. A validation error —
including a bind past loopback with no API key — is refused (exit 2) before
anything starts. A headless server launch runs **detached and persistent**
(`-d`, no `--rm`), unlike the GUI's foreground `--rm` server, so `--stop` and
`--health` can address it by container name.

Exit codes:

| code | `--launch` | `--stop` | `--health` |
|------|-----------|----------|-----------|
| 0 | started (ready, with `--wait`) | stopped / already stopped | ready |
| 1 | container run failed | stop failed | — |
| 2 | usage/config error | usage/config error | usage/config error |
| 3 | — | — | loading |
| 4 | — | — | down / stopped |
| 5 | `--wait` timed out (started, not ready) | — | — |

### JSON output

Add `--json` to any of the three commands to get one JSON object on stdout
(and nothing on stderr) instead of a human-readable line — for scripting and
test harnesses:

    llama-launcher --launch --profile ROUTER --json

Every outcome — success, warnings, action failure, and the pre-flight gate
refusal — is a single object. Warnings live inside the object, not on stderr.
The process exit code is unchanged (there is no `exit` field; `ok` mirrors it):

    {"action": "launch", "ok": true, "status": "started", "name": "llama-router",
     "host": "0.0.0.0", "port": 8080, "warnings": [], "error": null}

| field | meaning |
|-------|---------|
| `action` | `"launch"`, `"stop"`, or `"health"` |
| `ok` | `true` iff the process exit code is 0 |
| `status` | launch: `"started"` / `"ready"`; stop: `"stopped"`; health: `"ready"` / `"loading"` / raw state; `null` on failure |
| `name` / `host` / `port` | container name and address when known, else `null` |
| `warnings` | preset/router warnings (empty list when none) |
| `error` | failure or gate-refusal message, else `null` |

## Troubleshooting

### `crun: cannot stat /usr/lib/libnvidia-*.so.<version>: No such file or directory`

**Cause:** you're in CDI mode and the CDI spec is stale. The spec
(`/etc/cdi/nvidia.yaml`, or `~/.config/cdi/nvidia.yaml` for rootless) is a *static snapshot*:
it lists driver library paths pinned to an exact version **and** enumerates the specific GPUs
present when it was generated. It does not update itself. Anything that changes the driver files
or the set of cards will break it — most commonly:

- an **NVIDIA driver update** (the version-pinned `.so` paths no longer exist), or
- a **reboot that applied a pending driver update**, or
- **adding/removing a GPU** (the old spec still only knows about the previous cards).

**Fix — regenerate the spec on the host:**

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list        # confirm: nvidia.com/gpu=all, =0, =1, ...
```

Rootless podman without write access to `/etc/cdi`:

```bash
mkdir -p ~/.config/cdi
nvidia-ctk cdi generate --output ~/.config/cdi/nvidia.yaml
```

Then relaunch. After regenerating, `nvidia-ctk cdi list` should show **one entry per card**
(`=0`, `=1`, …); if a card is missing, `nvidia.com/gpu=all` won't use it. To pin a profile to a
single card, put e.g. `nvidia.com/gpu=1` in the launcher's extra run args instead of `all`.

> **Rule of thumb:** run `sudo nvidia-ctk cdi generate` any time the GPU hardware or the NVIDIA
> driver changes. As a temporary workaround you can switch the GPU mode to **Legacy — `--gpus all`**,
> which doesn't use the CDI spec — but regenerating the spec is the real fix.
