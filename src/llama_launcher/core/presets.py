"""Curated + user model-family presets, surfaced as one-click Suggestions.

A preset is a bundle of *suggested* catalog-setting values for a model family.
It is NOT applied automatically: preset_suggestions() turns it into the same
capabilities.Suggestion objects the UI already renders as click-to-apply chips,
so applying stays per-option and non-destructive (see MainWindow._apply_suggestion).

Pure module: no Qt, no IO. User-preset persistence lives in store/presets.py.
"""
from dataclasses import dataclass, field

from llama_launcher.core.capabilities import Suggestion


@dataclass(frozen=True)
class Preset:
    key: str                                    # stable id, e.g. "qwen3-moe"
    label: str                                  # display, e.g. "Qwen3-MoE"
    settings: dict = field(default_factory=dict)  # catalog-key -> suggested value
    source: str = "curated"                     # "curated" | "user"


def preset_suggestions(preset: Preset) -> list[Suggestion]:
    """One per-option Suggestion per setting, then one 'Apply all' Suggestion."""
    out = [Suggestion(text=f"{k} = {v}", settings={k: v}, fields={})
           for k, v in preset.settings.items()]
    out.append(Suggestion(text=f"Apply all {preset.label} defaults",
                          settings=dict(preset.settings), fields={}))
    return out


# Curated starter roster. Values are generic, catalog-valid, non-default
# suggestions the user is expected to tune; the guardrail test keeps them valid.
PRESETS: tuple[Preset, ...] = (
    Preset(key="qwen3-moe", label="Qwen3-MoE",
           settings={"temp": 0.6, "top-k": 20, "top-p": 0.8, "jinja": True}),
    Preset(key="general-chat", label="General chat",
           settings={"jinja": True, "flash-attn": "on"}),
)
