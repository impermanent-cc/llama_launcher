# llama_launcher: roadmap

Direction beyond the current cycle. Items move to SPEC.md when grilled and
settled, then to TASKS.md when a cycle picks them up.

## Next

- Review the remaining flags both engines accept that the catalog exposes
  for neither: the control-vector family (repeatable, one takes two tokens,
  needs panel plumbing like LoRA) and --spec-replace (two tokens). Decide
  per flag: add with plumbing, or record as out of scope.
- Periodic re-audit of both engines: common/arg.cpp against
  settings_catalog, root and ggml CMakeLists.txt against build_catalog.
- Decide how --fit (upstream default on) interacts with the app's own VRAM
  preflight, which may now be warning about sizes llama.cpp would shrink.
- Documentation cycle: README, CHANGELOG and RPC.md to ASCII and dash-free
  so they leave the guard allowlist; reword the legacy ` -- ` comment
  separators and turn the doubled-hyphen guard on.

## Later

- Live multi-node testing on a GPU worker; pooled inference across
  CPU-only rpc-servers crashes upstream and cannot be tested here.
- Server-mode --api-key delivered through a key file (mount plumbing; low
  severity on a single-user desktop).
- A model-file-existence warning before launch, routed so that it does not
  fire through the router's own health path.

## Not planned

- HF download flags and URL models: they bypass the local-path model and
  the GGUF preflight.
- CodeQL: costs Actions minutes for little gain here.
