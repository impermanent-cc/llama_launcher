# llama_launcher: specification

Settled requirements. Each numbered item is a decision the owner has agreed
to; grilling sessions append or amend items here before a plan is written.
Items are present tense and testable. Started fresh at the workflow stamp
(2026-09-02); earlier design records stay in project memory and are not
carried into this file.

## 1. Purpose

1.1 Llama Launcher turns a saved profile into a correct llama-server
command and runs it, in a container or natively, on one machine or across
remote nodes, with the state of the running server visible in the GUI.

## 2. Requirements

2.1 A profile is the unit of configuration: engine (mainline llama.cpp or
ik_llama.cpp), image or native binary, model path, GPU mode, and any
catalogued llama-server setting. `--dry-run --profile NAME` prints the
exact command without launching.

2.2 The settings catalog exposes only flags the chosen engine accepts;
mainline-only flags never reach an ik launch. Re-audits against upstream
`common/arg.cpp` and the CMake option lists are periodic maintenance.

2.3 Before a launch the app reads the GGUF header and estimates VRAM
against the selected GPU mode and context size, and warns or refuses
according to validation, rather than letting the server fail late.

2.4 Launch modes: container via podman or docker, native binary,
foreground in a detected terminal emulator, or headless control. The stop
grace period is configurable.

2.5 Remote nodes are reached over ssh and require podman on the node. An
RPC pool spreads a model across nodes' rpc-server workers; the RPC workers
table shows node, device and contribution. Pooled inference across
CPU-only workers is experimental and known to crash upstream.

2.6 The Monitor tab shows every instance as a card with logs and live
stats; the Stats dock shows CPU, GPU and memory; the Benchmark panel runs
prompt and generation sweeps and shows deltas against the previous run.

2.7 The router API key, LoRA adapter scales and model switching are driven
through the server's HTTP API, never by restarting the server.

2.8 One default port lives in core.spec.DEFAULT_PORT; no literal port
numbers elsewhere.

## 3. Constraints

3.1 Python 3.12 and 3.13 are the tested floor and ceiling; the code needs
only 3.10 but the declared floor is the CI-backed one.

3.2 No host paths, emails or secrets in tracked files; the repository is
public.

3.3 Qt runtime libraries used headless are listed in ci.yml and mirrored
in the localci Containerfile; the two lists stay in sync.

3.4 Tracked text is ASCII and free of em and en dashes except README.md,
CHANGELOG.md and RPC.md until their cleanup cycle. UI glyphs in code are
\u escapes.

## 4. Out of scope

- ROCm and AMD GPUs until a contributor with the hardware wires them up.
- HF download flags (--hf-repo, --hf-file, --model-url and kin): they cut
  against the mount-a-local-path model and bypass the VRAM preflight.
- Two-token flags the catalog cannot express (--spec-replace,
  --control-vector-layer-range) until they get panel plumbing like LoRA.
