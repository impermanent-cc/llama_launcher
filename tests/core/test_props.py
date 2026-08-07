from llama_launcher.core.props import PropsInfo, parse_props


_FULL = {
    "build_info": "b9755-0ef6f06d5",
    "total_slots": 2,
    "model_alias": "qwen35moe",
    "modalities": {"vision": True, "audio": False},
    "default_generation_settings": {"n_ctx": 8192, "params": {"temp": 1.0}},
}


def test_parses_full_props():
    info = parse_props(_FULL)
    assert info.build == "b9755-0ef6f06d5"
    assert info.n_ctx == 8192
    assert info.model_alias == "qwen35moe"
    assert info.total_slots == 2
    assert info.modalities == {"vision": True, "audio": False}


def test_n_ctx_falls_back_to_default_generation_settings():
    info = parse_props({"default_generation_settings": {"n_ctx": 4096}})
    assert info.n_ctx == 4096


def test_top_level_n_ctx_wins():
    info = parse_props({"n_ctx": 16384, "default_generation_settings": {"n_ctx": 4096}})
    assert info.n_ctx == 16384


def test_empty_props_is_all_none():
    info = parse_props({})
    assert info == PropsInfo(build=None, n_ctx=None, model_alias=None,
                             total_slots=None, modalities={})


def test_non_dict_is_safe():
    assert parse_props([]).n_ctx is None          # type: ignore[arg-type]
    assert parse_props(None).modalities == {}     # type: ignore[arg-type]


def test_modalities_keeps_only_booleans():
    info = parse_props({"modalities": {"vision": True, "junk": "nope", "n": 3}})
    assert info.modalities == {"vision": True}


def test_wrong_types_degrade_to_none():
    info = parse_props({"build_info": 123, "total_slots": "two", "n_ctx": "big"})
    assert info.build is None and info.total_slots is None and info.n_ctx is None
