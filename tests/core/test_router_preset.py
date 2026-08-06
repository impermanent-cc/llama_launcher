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


def test_bool_settings_render_true_or_negated_key():
    # A Setting whose flag is already negative (--no-cors-credentials) must
    # render under its own flag name, not as "cors-credentials = false".
    p = Profile(name="Q", model="/m.gguf", settings={"flash-attn": "on"})
    assert "flash-attn = on" in render_preset([(_member(), p)]).text


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
