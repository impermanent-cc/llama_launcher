from pathlib import Path

from llama_launcher.core.presets import Preset
from llama_launcher.store.presets import (
    save_preset, list_presets, delete_preset, preset_to_dict, preset_from_dict,
)


def test_dict_round_trip():
    p = Preset(key="fam", label="Fam", settings={"temp": 0.6}, source="user")
    assert preset_from_dict(preset_to_dict(p)) == p


def test_loaded_preset_defaults_to_user_source():
    # A stored preset with no explicit source is a user preset.
    p = preset_from_dict({"key": "fam", "label": "Fam", "settings": {"temp": 0.6}})
    assert p.source == "user"


def test_save_list_delete(tmp_path: Path):
    p = Preset(key="myfam", label="My Fam", settings={"top-k": 20}, source="user")
    save_preset(p, tmp_path)
    got = list_presets(tmp_path)
    assert [x.key for x in got] == ["myfam"]
    assert got[0].settings == {"top-k": 20} and got[0].source == "user"
    delete_preset("myfam", tmp_path)
    assert list_presets(tmp_path) == []


def test_list_empty_when_no_dir(tmp_path: Path):
    assert list_presets(tmp_path) == []


def test_list_skips_malformed_json_file(tmp_path: Path):
    p = Preset(key="good", label="Good", settings={"temp": 0.6}, source="user")
    save_preset(p, tmp_path)
    (tmp_path / "presets" / "broken.json").write_text("{ not valid json")
    got = list_presets(tmp_path)
    assert [x.key for x in got] == ["good"]


def test_list_skips_valid_json_missing_required_key(tmp_path: Path):
    p = Preset(key="good", label="Good", settings={"temp": 0.6}, source="user")
    save_preset(p, tmp_path)
    (tmp_path / "presets" / "nokey.json").write_text('{"label": "no key here"}')
    got = list_presets(tmp_path)
    assert [x.key for x in got] == ["good"]
