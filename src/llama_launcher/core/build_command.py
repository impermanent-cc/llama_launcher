import re
import shlex

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
    return [t for t in shlex.split(raw or "") if t.startswith("-D")]


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
            rendered = shlex.quote(str(value))
        out.append(f"-D{setting.flag}={rendered}")
    raw = parse_raw_defines(cfg.raw_defines)
    raw_names = {m.group(1) for d in raw if (m := _DEFINE_NAME.match(d))}
    out = [d for d in out
           if (m := _DEFINE_NAME.match(d)) and m.group(1) not in raw_names]
    return out + raw
