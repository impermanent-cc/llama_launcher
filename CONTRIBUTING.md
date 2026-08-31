# Contributing

Thanks for looking. Llama Launcher does what it set out to do, so the most
valuable contributions are bug reports, small fixes, and the two areas under
[Where help is wanted](#where-help-is-wanted).

## Open an issue first for anything large

For a typo, a crash fix, or a one function change, send the pull request
directly. For anything bigger, a new tab, a new launch mode, a new dependency, a
refactor, please open an issue first so the shape can be agreed. The app is
deliberately small and its scope is settled, so an unannounced large pull
request risks being declined after you have already done the work. An issue
costs you five minutes and can save you a weekend.

## Where help is wanted

- **AMD/ROCm support.** GPU probing is NVIDIA only, through `nvidia-smi`. The
  code lives in `src/llama_launcher/services/gpu.py` and needs a `rocm-smi`
  equivalent for GPU stats and VRAM fit estimation. Access to real AMD hardware
  to test on is the blocker here, not the code.
- **Upstream flag tracking.** llama.cpp and ik_llama.cpp add, rename, and retire
  server flags regularly. Keeping `core/settings_catalog.py` in step with
  upstream `common/arg.cpp`, and `core/build_catalog.py` in step with the root
  and ggml `CMakeLists.txt`, is ongoing work that is easy to pick up.

## Development setup

```bash
git clone https://github.com/impermanent-cc/llama_launcher.git
cd llama_launcher
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

`uv` works too; the README has the equivalent commands. Python 3.12 or newer is
required. You do not need podman, docker, a GPU, or a model file to run the
tests.

## Two constraints to know before writing code

**The settings catalog is the single source of truth.** A `Setting` entry in
`core/settings_catalog.py` produces both the form widget and the command line
argument. To support a new server flag, add the catalog entry; do not hand write
a widget in a panel or append a string in the command builder. The same holds
for `core/build_catalog.py` and the Build tab. This is why tracking upstream
flags is usually a small mechanical change.

**Tests must stay hermetic.** The suite runs in CI on a headless runner with no
podman, no docker, no GPU, and no network. `conftest.py` forces offscreen Qt and
points `XDG_CONFIG_HOME` at a temporary directory, so tests never touch your real
profiles. Never shell out to a real runtime in a test; stub the provider.
Anything blocking belongs off the UI thread, and long lived `QThread` objects are
avoided on purpose, since they abort on garbage collection across the suite. Use
a pooled `QRunnable` instead.

## Pull requests

- Keep the suite green with `pytest`. CI runs it on Python 3.12 and 3.13.
- Add tests for behavior you change. The suite is large and fast, and it is the
  main reason this project can be refactored with any confidence.
- Match the surrounding style rather than reformatting. There is no
  autoformatter in CI.
- Commit subjects follow `type(scope): summary` in lower case, for example
  `fix(build): scope podman build context to the Containerfile's own dir`.
- One logical change per pull request.

## Changelog and releases

User-visible changes go in `CHANGELOG.md` under `## [Unreleased]`, in the same
pull request as the change. Internal refactors and test-only work don't need an
entry.

Each released section is the single source for that release's GitHub notes, so
release notes are never written by hand on the release page:

```bash
# Rename [Unreleased] to the new version, add the date, then:
git tag -a v0.2.0 -m "llama_launcher v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --notes-file <(./scripts/release-notes.sh 0.2.0)
```

To correct notes after publishing, edit `CHANGELOG.md` and re-run the same
extraction with `gh release edit v0.2.0 --notes-file ...`. A test asserts the
version in `pyproject.toml` always has a section it can publish, so bump both
together.

## Reporting bugs

Use the bug report template. The most useful attachment is the output of the
**Generate report** button in the app, which collects the command, profile,
validation state, runtime and GPU detail, metrics, and recent logs, with API keys
redacted. It does not redact host paths, so glance over it before pasting if your
directory names are private.

## Security

Please do not report vulnerabilities in a public issue. See
[SECURITY.md](SECURITY.md).
