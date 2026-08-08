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
