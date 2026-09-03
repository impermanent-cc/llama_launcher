from dataclasses import asdict, dataclass, field, fields


@dataclass
class BuildConfig:
    name: str = ""
    engine: str = "llama.cpp"  # "llama.cpp" | "ik_llama.cpp"
    target: str = "native"  # "native" | "container"
    git_ref: str = ""  # "" = engine's default branch
    source_dir: str = ""  # native target: user's checkout
    builder_image: str = ""  # container target
    runtime_image: str = ""
    tag_override: str = ""
    options: dict = field(default_factory=dict)  # non-default catalog values
    raw_defines: str = ""  # extra -D defines, raw-args-style escape hatch


@dataclass
class BuildOutput:
    id: str
    kind: str  # "tag" | "binary"
    identifier: str  # image tag, or absolute binary path
    config_name: str
    engine: str
    git_ref: str
    options: dict
    created: str  # ISO date
    notes: str = ""


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def build_config_to_dict(c: BuildConfig) -> dict:
    return asdict(c)


def build_config_from_dict(d: dict) -> BuildConfig:
    return _from_dict(BuildConfig, d)


def build_output_to_dict(o: BuildOutput) -> dict:
    return asdict(o)


def build_output_from_dict(d: dict) -> BuildOutput:
    return _from_dict(BuildOutput, d)
