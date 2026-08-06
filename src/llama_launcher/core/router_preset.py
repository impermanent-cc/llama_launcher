"""Render llama.cpp router preset INI files from saved profiles.

The router (`llama-server` started with no model) reads an INI where each
section is a model id and each key is a llama-server argument without its
leading dashes. See tools/server/README.md, "Model presets".

Pure module: no IO, no Qt.
"""

import shlex
from dataclasses import dataclass, field

from .settings_catalog import CATALOG, ROUTER_ONLY_KEYS
from .spec import Profile, RouterMember, member_model_id

# Keys the router owns. llama.cpp strips or overwrites these when it launches a
# child instance, so emitting them is at best noise and at worst confusing.
EXCLUDED_PRESET_KEYS: frozenset = frozenset({
    "port", "host", "api-key", "alias",
}) | ROUTER_ONLY_KEYS


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
        if not tok.startswith("-"):
            problems.append(f"cannot express {tok!r} in a preset (not a --flag)")
            i += 1
            continue
        key = tok.lstrip("-")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
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
    out: list[tuple[str, str]] = []
    for key, setting in catalog.items():
        if key in EXCLUDED_PRESET_KEYS or key not in profile.settings:
            continue
        value = profile.settings[key]
        # The INI key is the flag itself, minus dashes — including negative
        # flags such as --no-cors-credentials, whose key is "no-cors-credentials".
        ini_key = setting.flag.lstrip("-")
        if setting.type == "bool":
            if not value:
                continue
            out.append((ini_key, "true"))
        else:
            out.append((ini_key, str(value)))
    return out


def render_preset(pairs: list, catalog: dict = CATALOG) -> PresetResult:
    """Render `[(RouterMember, Profile), ...]` into preset INI text."""
    lines: list[str] = ["version = 1", ""]
    warnings: list[str] = []

    for member, profile in pairs:
        model_id = member_model_id(member)
        lines.append(f"[{model_id}]")

        if profile.model:
            lines.append(f"model = {profile.model}")
        if profile.mmproj:
            lines.append(f"mmproj = {profile.mmproj}")
        if profile.draft_model:
            lines.append(f"spec-draft-model = {profile.draft_model}")

        if profile.loras:
            lines.append(f"lora = {profile.loras[0].path}")
            if len(profile.loras) > 1:
                dropped = ", ".join(l.path for l in profile.loras[1:])
                warnings.append(
                    f"{profile.name}: more than one LoRA cannot be expressed in a "
                    f"preset (INI keys are unique); dropped: {dropped}"
                )

        for key, value in _setting_pairs(profile, catalog):
            lines.append(f"{key} = {value}")

        raw_pairs, problems = convert_raw_args(profile.raw_args)
        for key, value in raw_pairs.items():
            if key in EXCLUDED_PRESET_KEYS:
                continue
            lines.append(f"{key} = {value}")
        for problem in problems:
            warnings.append(f"{profile.name}: {problem}")

        lines.append(f"load-on-startup = {'true' if member.load_on_startup else 'false'}")
        lines.append(f"stop-timeout = {member.stop_timeout}")
        lines.append("")

    return PresetResult(text="\n".join(lines), warnings=warnings)
