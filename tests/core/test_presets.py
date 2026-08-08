from llama_launcher.core.capabilities import Suggestion
from llama_launcher.core.presets import Preset, preset_suggestions

_P = Preset(key="fam", label="Fam", settings={"temp": 0.6, "top-k": 20})


def test_one_suggestion_per_option_plus_apply_all():
    sgs = preset_suggestions(_P)
    # one per option (order-independent) ...
    per_option = {s.text: s for s in sgs if s.text != "Apply all Fam defaults"}
    assert per_option["temp = 0.6"].settings == {"temp": 0.6}
    assert per_option["temp = 0.6"].fields == {}
    assert per_option["top-k = 20"].settings == {"top-k": 20}
    # ... plus exactly one bulk suggestion carrying every option, appended last.
    assert sgs[-1].text == "Apply all Fam defaults"
    assert sgs[-1].settings == {"temp": 0.6, "top-k": 20}
    assert len(sgs) == 3


def test_suggestion_dicts_are_copies():
    sgs = preset_suggestions(_P)
    sgs[-1].settings["temp"] = 999
    assert _P.settings["temp"] == 0.6           # preset not mutated/aliased


def test_empty_preset_yields_only_apply_all_or_nothing():
    # A preset with no settings has no per-option chips; the bulk chip carries {}.
    sgs = preset_suggestions(Preset(key="e", label="E", settings={}))
    assert [s.text for s in sgs] == ["Apply all E defaults"]
    assert sgs[0].settings == {}


from llama_launcher.core.presets import PRESETS
from llama_launcher.core.settings_catalog import CATALOG


def _valid_for(setting, val) -> bool:
    t = setting.type
    if t == "bool":
        return isinstance(val, bool)
    if t == "enum":
        return val in setting.enum
    if t in ("int", "float"):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return False
        if setting.minimum is not None and val < setting.minimum:
            return False
        if setting.maximum is not None and val > setting.maximum:
            return False
        return True
    if t == "int_or_token":
        return val in setting.tokens or isinstance(val, int)
    if t == "string":
        return isinstance(val, str)
    return False


def test_presets_roster_nonempty_with_unique_keys():
    assert PRESETS
    keys = [p.key for p in PRESETS]
    assert len(keys) == len(set(keys))


def test_curated_presets_valid_against_catalog():
    for preset in PRESETS:
        assert preset.settings, f"{preset.key} has no settings (dead preset)"
        for key, val in preset.settings.items():
            assert key in CATALOG, f"{preset.key}: unknown setting {key}"
            s = CATALOG[key]
            assert _valid_for(s, val), f"{preset.key}: {key}={val!r} invalid for {s.type}"
            assert val != s.default, f"{preset.key}: {key}={val!r} equals default (dead-weight)"
