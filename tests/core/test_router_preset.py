from llama_launcher.core.router_preset import (
    convert_raw_args, render_preset,
)
from llama_launcher.core.spec import LoraRef, Profile, RouterMember


def _member(name="Qwen", **kw):
    return RouterMember(profile=name, **kw)


def test_renders_version_header_and_section_per_member():
    p = Profile(name="Qwen", model="/models/qwen.gguf")
    out = render_preset([(_member(), p)])
    assert out.text.startswith("version = 1\n")
    assert "[qwen]" in out.text
    assert "model = /models/qwen.gguf" in out.text


def test_section_name_uses_explicit_model_id():
    p = Profile(name="Qwen", model="/models/qwen.gguf")
    out = render_preset([(_member(model_id="big-moe"), p)])
    assert "[big-moe]" in out.text
    assert "[qwen]" not in out.text


def test_renders_settings_as_dashless_keys():
    p = Profile(name="Qwen", model="/m.gguf",
                settings={"ctx-size": 8192, "n-gpu-layers": 99})
    text = render_preset([(_member(), p)]).text
    assert "ctx-size = 8192" in text
    assert "n-gpu-layers = 99" in text


def test_enum_settings_render_their_value():
    p = Profile(name="Q", model="/m.gguf", settings={"flash-attn": "on"})
    assert "flash-attn = on" in render_preset([(_member(), p)]).text


def test_bool_settings_render_true_under_their_own_flag_name():
    # A Setting whose flag is already negative (--no-cors-credentials) must
    # render under its own flag name, not as "cors-credentials = true".
    p = Profile(name="Q", model="/m.gguf", settings={"cors-credentials": True})
    text = render_preset([(_member(), p)]).text
    assert "no-cors-credentials = true" in text


def test_false_bool_settings_emit_nothing():
    p = Profile(name="Q", model="/m.gguf", settings={"cors-credentials": False})
    assert "cors-credentials" not in render_preset([(_member(), p)]).text


def test_raw_arg_duplicating_a_form_setting_is_reported():
    p = Profile(name="Q", model="/m.gguf", settings={"ctx-size": 8192},
                raw_args="--ctx-size 4096")
    out = render_preset([(_member(), p)])
    assert out.text.count("ctx-size = ") == 1
    assert "ctx-size = 8192" in out.text
    assert any("duplicates" in w for w in out.warnings)


def test_excludes_router_controlled_keys():
    p = Profile(name="Q", model="/m.gguf",
                settings={"port": 9090, "api-key": "secret", "ctx-size": 4096})
    text = render_preset([(_member(), p)]).text
    assert "port" not in text
    assert "secret" not in text
    assert "ctx-size = 4096" in text


def test_excludes_router_only_keys():
    p = Profile(name="Q", model="/m.gguf", settings={"models-max": 4})
    assert "models-max" not in render_preset([(_member(), p)]).text


def test_emits_mmproj_and_draft_model():
    p = Profile(name="Q", model="/m.gguf", mmproj="/mm.gguf",
                draft_model="/d.gguf")
    text = render_preset([(_member(), p)]).text
    assert "mmproj = /mm.gguf" in text
    assert "spec-draft-model = /d.gguf" in text


def test_emits_preset_only_keys():
    p = Profile(name="Q", model="/m.gguf")
    text = render_preset([(_member(load_on_startup=True, stop_timeout=30), p)]).text
    assert "load-on-startup = true" in text
    assert "stop-timeout = 30" in text


def test_single_lora_emitted_multiple_warns():
    p1 = Profile(name="Q", model="/m.gguf", loras=[LoraRef(path="/a.gguf")])
    out1 = render_preset([(_member(), p1)])
    assert "lora = /a.gguf" in out1.text
    assert out1.warnings == []

    p2 = Profile(name="Q", model="/m.gguf",
                 loras=[LoraRef(path="/a.gguf"), LoraRef(path="/b.gguf")])
    out2 = render_preset([(_member(), p2)])
    # INI keys are unique, so only the first survives; the user must be told.
    assert "lora = /a.gguf" in out2.text
    assert "/b.gguf" not in out2.text
    assert any("more than one LoRA" in w for w in out2.warnings)


def test_raw_args_converted_to_keys():
    pairs, problems = convert_raw_args("--foo bar --flag")
    assert pairs == {"foo": "bar", "flag": "true"}
    assert problems == []


def test_raw_args_repeated_key_reported():
    pairs, problems = convert_raw_args("--foo a --foo b")
    assert pairs == {"foo": "a"}
    assert any("foo" in p for p in problems)


def test_raw_args_positional_reported():
    pairs, problems = convert_raw_args("bare-token")
    assert pairs == {}
    assert any("bare-token" in p for p in problems)


def test_raw_args_problems_surface_as_warnings():
    p = Profile(name="Q", model="/m.gguf", raw_args="--foo a --foo b")
    out = render_preset([(_member(), p)])
    assert any("Q" in w for w in out.warnings)


def test_multiple_members_each_get_a_section():
    a = Profile(name="A", model="/a.gguf")
    b = Profile(name="B", model="/b.gguf")
    text = render_preset([(_member("A"), a), (_member("B"), b)]).text
    assert "[a]" in text and "[b]" in text


def test_raw_args_negative_values_are_not_mistaken_for_flags():
    # llama.cpp uses -1 sentinels widely; "-1" must parse as a value, not as
    # a flag that turns the real flag into "true" and emits a junk "1 = true".
    pairs, problems = convert_raw_args("--temp -1")
    assert pairs == {"temp": "-1"}
    assert problems == []


def test_raw_args_multiple_negative_values():
    pairs, problems = convert_raw_args("--seed -1 --top-k 40")
    assert pairs == {"seed": "-1", "top-k": "40"}
    assert problems == []


def test_raw_args_short_flag_with_negative_value():
    pairs, _ = convert_raw_args("-ngl -1")
    assert pairs == {"ngl": "-1"}


def test_raw_args_equals_form_is_split():
    pairs, problems = convert_raw_args("--ctx-size=4096")
    assert pairs == {"ctx-size": "4096"}
    assert problems == []


def test_raw_args_negative_float_value():
    pairs, _ = convert_raw_args("--top-n-sigma -1.5")
    assert pairs == {"top-n-sigma": "-1.5"}


def test_raw_args_still_treats_a_real_flag_as_a_flag():
    pairs, _ = convert_raw_args("--flag --other x")
    assert pairs == {"flag": "true", "other": "x"}


def test_preset_omits_engine_gated_flag_for_mainline_member():
    # An ik_llama.cpp-only flag on a llama.cpp member must not reach a mainline
    # router's preset (the child llama-server would reject it) -- parity with
    # command_builder._owned_server_pairs.
    from llama_launcher.core.spec import Runtime
    p = Profile(name="Q", model="/m.gguf", settings={"mla-use": "1"},
                runtime=Runtime(engine="llama.cpp"))
    assert "mla-use" not in render_preset([(_member(), p)]).text


def test_preset_omits_enum_value_equal_to_default():
    # An enum left at its default is a "leave engine default" sentinel; emitting
    # it is redundant (and for ik's mla-use 'auto', invalid).
    p = Profile(name="Q", model="/m.gguf", settings={"flash-attn": "auto"})
    assert "flash-attn" not in render_preset([(_member(), p)]).text


def test_preset_translates_spec_type_for_ik():
    from llama_launcher.core.spec import Runtime
    p = Profile(name="Q", model="/m.gguf", settings={"spec-type": "draft-mtp"},
                runtime=Runtime(engine="ik_llama.cpp"))
    assert "spec-type = mtp" in render_preset([(_member(), p)]).text


def test_preset_drops_suffix_spec_type_on_mainline():
    from llama_launcher.core.spec import Runtime
    p = Profile(name="Q", model="/m.gguf", settings={"spec-type": "suffix"},
                runtime=Runtime(engine="llama.cpp"))
    assert "spec-type" not in render_preset([(_member(), p)]).text


def test_preset_load_mode_at_default_does_not_suppress_legacy():
    # Parity with command_builder: a leftover load-mode equal to its default
    # emits nothing, so it must not eat the legacy no-mmap/mlock either.
    p = Profile(name="Q", model="/m.gguf",
                settings={"load-mode": "auto", "no-mmap": True})
    text = render_preset([(_member(), p)]).text
    assert "load-mode" not in text
    assert "no-mmap = true" in text


def test_preset_omits_blank_string_value():
    p = Profile(name="Q", model="/m.gguf", settings={"tensor-split": ""})
    assert "tensor-split" not in render_preset([(_member(), p)]).text


def test_preset_rejects_newline_injection_in_path_field():
    # A newline in a path field must not inject arbitrary preset keys/sections.
    evil = "/m.gguf\nload-on-startup = true\n[injected]\nmodel = /etc/shadow"
    p = Profile(name="Q", model=evil)
    res = render_preset([(_member(), p)])
    assert "[injected]" not in res.text
    assert "/etc/shadow" not in res.text
    assert res.warnings  # dropped-with-warning, not silently emitted
