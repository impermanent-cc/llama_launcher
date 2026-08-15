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

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv                                # create .venv (Python ≥ 3.13)
uv pip install -e ".[dev]"

uv run llama-launcher                  # GUI  (NOT `llama_launcher.app` -- that's a module, not a command)
uv run python -m llama_launcher.app    # module form of the same GUI
uv run python -m llama_launcher.app --dry-run --profile "NAME"
uv run pytest -q                       # test suite
```

## GPU passthrough

The launcher offers two GPU modes (Configure tab → **Runtime**):

- **CDI — `--device nvidia.com/gpu=all`** (recommended) — uses the NVIDIA Container Device
  Interface spec.
- **Legacy — `--gpus all`** — the older runtime hook; doesn't read the CDI spec.

### Run detached (no terminal window)

Each server profile has a **Run detached** checkbox next to Launch. Leave it
off (the default) to launch into a terminal window as before. Check it to
launch with no terminal — the container runs detached (`-d`), its output
streams to the **Monitor** tab, and **Stop** shuts it down. If the launch
fails (bad image, GPU/CDI, bad flag), a dialog shows the failure reason; the
container also persists (it isn't auto-removed), so its logs remain readable
after a crash. The setting is saved per profile. Router profiles are always
detached, so the checkbox is hidden for them.

### ik_llama.cpp engine

Select **Engine → ik_llama.cpp** in Configure to run ikawrakow's fork instead of
mainline llama.cpp. It uses the same `llama-server` CLI, so profiles work the same
way; the launcher adds ik-only flags (run-time-repack, MLA, fused-MoE toggle,
attention-max-batch, smart-expert-reduction) and extra KV-cache quant types, shown
only when this engine is selected and never passed to a mainline launch.

Images live at `ghcr.io/ikawrakow/ik-llama-cpp` — use the `*-server` tag for single
server mode and the `*-swap` tag for router mode (e.g. `cu12-server`, `cpu-server`).
**Detect** lists your locally-pulled ik images when this engine is selected.

Self-built (native, non-container) launch is not yet supported for either engine —
run via Podman or Docker for now.

## Family presets

Next to the settings is a **Suggest for family** picker. Choose a family
(curated, e.g. Qwen3-MoE, or one you saved) and its recommended flags appear as
one-click 💡 chips in the suggestions strip — the same chips the app already
shows for GGUF-detected capabilities. Each chip is a **suggestion, not a rule**:
nothing changes until you click it, and clicking applies only that one option
(there's also an "Apply all …" chip). Your model, mounts, and any values you
don't accept are left untouched.

**Save as preset…** turns the options you've currently set into a named preset
that then appears in the picker. Presets store only `llama-server` settings —
never model paths — and a preset you save overrides a curated one of the same
name.

## Instances panel (Monitor tab)

The top of the Monitor tab lists every server this launcher has started, with
its port, health, and a live summary stat (generation tok/s, or "ready" for an
embedding/rerank server). This means you can run, say, a generation model **and**
an embedding model for RAG at once and keep an eye on both — switching the
Configure form to another profile no longer hides what's running.

Click a row to point the full Monitor (and logs) below at that instance; the
■ button on a row stops it. Selecting an instance to watch does not change the
profile you're editing.

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

## Router API key

A router authenticates clients with a bearer API key (`sk-…`), shown on the
Router panel with Reveal/Copy. Point your harness at `http://HOST:PORT/v1`
with this key.

Two scopes control where the key comes from:

- **Global** (default) — one shared key used by every router profile. Set it
  once and every global-scope profile serves it, so a harness needs the key
  only once, even when you switch profiles.
- **Own key for this profile** — pin a distinct key on a specific profile
  when you want it isolated from the shared one.

Use **Edit…** to paste your own key value or click **Generate** for a fresh
one, then **Save**. Saving a global key updates every global-scope profile at
once. Changing a key takes effect the next time the router launches.

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
