# Llama Launcher

A PySide6/Qt6 desktop app that builds a `podman`/`docker` `llama-server` command from a GUI
and launches it in a terminal (konsole, foreground so `Ctrl-C` works), then observes the named
container from outside. Tabbed **Configure / Monitor** UI with profiles, a curated `llama-server`
settings catalog (plus a raw-args escape hatch), typed mounts, mmproj/LoRA/draft-model pickers,
model-aware capability detection, VRAM preflight, and a live throughput/MTP monitor.

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
