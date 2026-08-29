"""Render llama.cpp router preset INI files from saved profiles.

The router (`llama-server` started with no model) reads an INI where each
section is a model id and each key is a llama-server argument without its
leading dashes. See tools/server/README.md, "Model presets".

Pure module: no IO, no Qt.
"""

import re
import shlex
from dataclasses import dataclass, field

from .settings_catalog import (
    CATALOG, ROUTER_ONLY_KEYS,
    IK_EXTRA_KV_CACHE_TYPES, IK_EXTRA_SPEC_TYPES, IK_SPEC_TYPE_RENAMES,
)
from .spec import Profile, RouterMember, member_model_id

# Keys the router owns. llama.cpp strips or overwrites these when it launches a
# child instance, so emitting them is at best noise and at worst confusing.
EXCLUDED_PRESET_KEYS: frozenset = frozenset({
    "port", "host", "api-key", "alias",
}) | ROUTER_ONLY_KEYS

# A flag starts with one or two dashes followed by a LETTER. "-1" and "-1.5" are
# values, not flags; llama.cpp uses negative sentinels all over its interface
# (--seed -1, --n-gpu-layers -1, --top-n-sigma -1.5).
_FLAG_RE = re.compile(r"^--?[A-Za-z]")


def _is_flag(token: str) -> bool:
    return bool(_FLAG_RE.match(token))


@dataclass
class PresetResult:
    text: str
    warnings: list[str] = field(default_factory=list)


def convert_raw_args(raw: str) -> tuple[dict, list[str]]:
    """Convert a free-form argument string into INI key/value pairs.

    Returns (pairs, problems). A flag with no value becomes "true". Anything
    that repeats a key or is a bare positional is reported rather than guessed
    at, because an INI cannot express it.
    """
    pairs: dict = {}
    problems: list[str] = []
    if not raw.strip():
        return pairs, problems

    tokens = shlex.split(raw)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not _is_flag(tok):
            problems.append(f"cannot express {tok!r} in a preset (not a --flag)")
            i += 1
            continue
        # --key=value is as valid as --key value.
        if "=" in tok:
            key, _, value = tok.partition("=")
            key = key.lstrip("-")
            i += 1
        else:
            key = tok.lstrip("-")
            if i + 1 < len(tokens) and not _is_flag(tokens[i + 1]):
                value = tokens[i + 1]
                i += 2
            else:
                value = "true"
                i += 1
        if key in pairs:
            problems.append(f"{key!r} appears more than once; INI keys are unique")
            continue
        pairs[key] = value
    return pairs, problems


def _setting_pairs(profile: Profile, catalog: dict) -> list[tuple[str, str]]:
    """Preset key/value pairs for a member's settings.

    Applies the SAME gating as command_builder._owned_server_pairs so the two
    argv-generation paths agree: engine-gated flags never reach a mismatched
    engine, ik-only KV-cache VALUES are dropped on a mainline launch, an enum
    left at its default (a "leave engine default" sentinel) is skipped, blanks
    emit nothing, and --load-mode supersedes the legacy --no-mmap/--mlock. A
    router that skipped these emitted flags the child llama-server then rejects.
    """
    out: list[tuple[str, str]] = []
    engine = profile.runtime.engine
    # Parity with command_builder: suppress the legacy pair only when load-mode
    # will actually emit (a value at its default is skipped as an enum sentinel).
    _lm_default = CATALOG["load-mode"].default
    suppress = ({"no-mmap", "mlock"}
                if profile.settings.get("load-mode", _lm_default) != _lm_default
                else set())
    for key, setting in catalog.items():
        if key in EXCLUDED_PRESET_KEYS or key in suppress or key not in profile.settings:
            continue
        if setting.engine != "any" and setting.engine != engine:
            continue
        value = profile.settings[key]
        if (key in ("cache-type-k", "cache-type-v")
                and value in IK_EXTRA_KV_CACHE_TYPES and engine != "ik_llama.cpp"):
            continue
        # Same layering for spec-type: drop ik-only values on mainline, rename
        # the shared draft-* spellings to ik's un-prefixed ones on ik.
        if key == "spec-type":
            if value in IK_EXTRA_SPEC_TYPES and engine != "ik_llama.cpp":
                continue
            if engine == "ik_llama.cpp":
                value = IK_SPEC_TYPE_RENAMES.get(value, value)
        if setting.type == "enum" and value == setting.default:
            continue
        # The INI key is the flag itself, minus dashes, including negative
        # flags such as --no-cors-credentials, whose key is "no-cors-credentials".
        ini_key = setting.flag.lstrip("-")
        if setting.type == "bool":
            if not value:
                continue
            out.append((ini_key, "true"))
        else:
            if not str(value).strip():          # blank -> emit nothing
                continue
            out.append((ini_key, str(value)))
    return out


def _ini_safe(value: str) -> bool:
    """An INI value must not carry a newline: a newline in a path or setting
    value would inject arbitrary preset keys/sections (the section is served to
    the child llama-server). CR is rejected too (some parsers split on it)."""
    return "\n" not in value and "\r" not in value


def render_preset(pairs: list, catalog: dict = CATALOG) -> PresetResult:
    """Render `[(RouterMember, Profile), ...]` into preset INI text."""
    lines: list[str] = ["version = 1", ""]
    warnings: list[str] = []

    for member, profile in pairs:
        def emit(key: str, value) -> bool:
            """Append `key = value`, dropping (with a warning) any value that
            carries a newline so it cannot inject preset keys/sections."""
            if not _ini_safe(str(value)):
                warnings.append(
                    f"{profile.name}: {key!r} value contains a newline and was "
                    f"dropped (it would inject router preset keys).")
                return False
            lines.append(f"{key} = {value}")
            return True

        model_id = member_model_id(member)
        if not _ini_safe(model_id):
            warnings.append(f"{profile.name}: model id contains a newline; member skipped.")
            continue
        lines.append(f"[{model_id}]")

        if profile.model:
            emit("model", profile.model)
        if profile.mmproj:
            emit("mmproj", profile.mmproj)
        if profile.draft_model:
            emit("spec-draft-model", profile.draft_model)

        if profile.loras:
            emit("lora", profile.loras[0].path)
            if len(profile.loras) > 1:
                dropped = ", ".join(l.path for l in profile.loras[1:])
                warnings.append(
                    f"{profile.name}: more than one LoRA cannot be expressed in a "
                    f"preset (INI keys are unique); dropped: {dropped}"
                )

        emitted: set = set()
        for key, value in _setting_pairs(profile, catalog):
            if emit(key, value):
                emitted.add(key)

        raw_pairs, problems = convert_raw_args(profile.raw_args)
        for key, value in raw_pairs.items():
            if key in EXCLUDED_PRESET_KEYS:
                continue
            # INI keys are unique; a raw arg repeating a key already set in the
            # form would silently depend on the parser's last-wins behaviour.
            if key in emitted:
                problems.append(
                    f"raw arg {key!r} duplicates a value already set in the form; "
                    f"the form value is kept")
                continue
            if emit(key, value):
                emitted.add(key)
        for problem in problems:
            warnings.append(f"{profile.name}: {problem}")

        lines.append(f"load-on-startup = {'true' if member.load_on_startup else 'false'}")
        lines.append(f"stop-timeout = {member.stop_timeout}")
        lines.append("")

    return PresetResult(text="\n".join(lines), warnings=warnings)
