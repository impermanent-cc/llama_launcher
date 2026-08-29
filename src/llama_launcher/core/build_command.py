import posixpath
import re
import shlex
from dataclasses import dataclass

from .build_catalog import BUILD_CATALOG, DEFAULT_BRANCH, ENGINE_SHORT, REPO_URL
from .build_spec import BuildConfig
from .settings_catalog import for_engine


def config_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "config"


def auto_tag(cfg: BuildConfig, existing: set, today) -> str:
    if cfg.tag_override:
        return cfg.tag_override
    base = f"{ENGINE_SHORT[cfg.engine]}-custom:{config_slug(cfg.name)}-{today:%Y%m%d}"
    tag, n = base, 1
    while tag in existing:
        n += 1
        tag = f"{base}-{n}"
    return tag


def parse_raw_defines(raw: str) -> list[str]:
    # Fires per keystroke from the Raw defines field: a half-typed quote makes
    # shlex raise ValueError, which must not escape into the Qt slot. Degrade
    # to whitespace splitting until the quote is closed.
    try:
        tokens = shlex.split(raw or "")
    except ValueError:
        tokens = (raw or "").split()
    return [t for t in tokens if t.startswith("-D")]


_RPC_DEFINE = re.compile(r"^-DGGML_RPC(?::[A-Za-z]+)?=(.*)$")


def _rpc_enabled(defines: list[str]) -> bool:
    """Whether the rendered defines turn GGML_RPC on, accepting every CMake
    truthy spelling (ON/1/TRUE/YES, any case, optional :BOOL type suffix) --
    a raw '-DGGML_RPC=1' must still build the rpc-server target."""
    enabled = False
    for d in defines:
        m = _RPC_DEFINE.match(d)
        if m:
            enabled = m.group(1).strip("'\"").upper() in ("ON", "1", "TRUE", "YES")
    return enabled


_DEFINE_NAME = re.compile(r"^-D([A-Za-z0-9_]+)")


def render_defines(cfg: BuildConfig) -> list[str]:
    cat = for_engine(BUILD_CATALOG, cfg.engine)
    out: list[str] = []
    for key, setting in cat.items():
        if key not in cfg.options:
            continue
        value = cfg.options[key]
        if value == setting.default:
            continue
        if setting.type == "bool":
            rendered = "ON" if value else "OFF"
        else:
            # A cleared field is "leave upstream alone", never -DNAME='' --
            # same blank guard as command_builder._render_setting.
            if not str(value).strip():
                continue
            rendered = shlex.quote(str(value))
        out.append(f"-D{setting.flag}={rendered}")
    raw = parse_raw_defines(cfg.raw_defines)
    raw_names = {m.group(1) for d in raw if (m := _DEFINE_NAME.match(d))}
    out = [d for d in out
           if (m := _DEFINE_NAME.match(d)) and m.group(1) not in raw_names]
    return out + raw


@dataclass
class NativeBuild:
    configure_cmd: str
    build_cmd: str
    expected_binary: str


def render_native(cfg: BuildConfig) -> NativeBuild:
    slug = config_slug(cfg.name)
    build_dir = f"build-{slug}"
    defines = render_defines(cfg)
    targets = "llama-server"
    if _rpc_enabled(defines):
        targets += " rpc-server"
    configure = " ".join(["cmake", "-B", build_dir, *defines])
    build = f"cmake --build {build_dir} -j$(nproc) --target {targets}"
    binary = f"{cfg.source_dir.rstrip('/')}/{build_dir}/bin/llama-server"
    return NativeBuild(configure, build, binary)


@dataclass
class ContainerBuild:
    containerfile: str
    build_cmd: str


_CONTAINERFILE = """\
FROM {builder} AS build
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git cmake build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone {repo} src
RUN git -C src checkout {ref}
WORKDIR /opt/src
RUN cmake -B build {defines} && \\
    cmake --build build -j$(nproc) --target {targets}
FROM {runtime}
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/src/build/bin/ /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
"""


def default_images(cfg: BuildConfig) -> tuple:
    if cfg.options.get("cuda"):
        return ("docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04",
                "docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04")
    return ("docker.io/library/debian:bookworm",
            "docker.io/library/debian:bookworm-slim")


def render_container(cfg: BuildConfig, tag: str,
                     containerfile_path: str) -> ContainerBuild:
    defines = render_defines(cfg)
    targets = "llama-server"
    if _rpc_enabled(defines):
        targets += " rpc-server"
    cf = _CONTAINERFILE.format(
        builder=cfg.builder_image,
        runtime=cfg.runtime_image,
        repo=REPO_URL[cfg.engine],
        ref=cfg.git_ref or DEFAULT_BRANCH[cfg.engine],
        defines=" ".join(defines),
        targets=targets,
    )
    # Build context is the Containerfile's own parent dir, not the caller's
    # CWD: the Containerfile clones its own source from REPO_URL, so tarring
    # an unrelated CWD as build context is both wrong and wasteful.
    context_dir = posixpath.dirname(containerfile_path) or "."
    return ContainerBuild(
        cf, f"podman build -t {tag} -f {containerfile_path} {context_dir}")
