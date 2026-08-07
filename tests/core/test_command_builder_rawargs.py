from llama_launcher.core import command_builder as cb


def test_canonical_folds_catalog_alias_to_long_form():
    assert cb._canonical_flag("-ngl") == "--n-gpu-layers"
    assert cb._canonical_flag("-c") == "--ctx-size"
    assert cb._canonical_flag("-fa") == "--flash-attn"


def test_canonical_long_form_is_identity():
    assert cb._canonical_flag("--n-gpu-layers") == "--n-gpu-layers"
    assert cb._canonical_flag("--ctx-size") == "--ctx-size"


def test_canonical_structural_flags():
    assert cb._canonical_flag("-m") == "--model"
    assert cb._canonical_flag("--model") == "--model"
    assert cb._canonical_flag("--port") == "--port"
    assert cb._canonical_flag("--host") == "--host"


def test_canonical_unknown_flag_is_identity():
    assert cb._canonical_flag("--numa") == "--numa"
    assert cb._canonical_flag("--totally-made-up") == "--totally-made-up"


def test_parse_flag_value_pair():
    assert cb._parse_raw_pairs("-ngl 50") == [("-ngl", "50")]


def test_parse_equals_form():
    assert cb._parse_raw_pairs("--ctx-size=8192") == [("--ctx-size", "8192")]


def test_parse_bare_flag_is_none_value():
    assert cb._parse_raw_pairs("--mlock") == [("--mlock", None)]
    assert cb._parse_raw_pairs("--mlock --no-mmap") == [("--mlock", None), ("--no-mmap", None)]


def test_parse_negative_value_is_not_a_flag():
    assert cb._parse_raw_pairs("-ngl -1") == [("-ngl", "-1")]
    assert cb._parse_raw_pairs("--top-n-sigma -1.5") == [("--top-n-sigma", "-1.5")]


def test_parse_preserves_order_and_repeats():
    assert cb._parse_raw_pairs("--lora /a.gguf --lora /b.gguf") == [
        ("--lora", "/a.gguf"), ("--lora", "/b.gguf")]


def test_parse_empty_is_empty():
    assert cb._parse_raw_pairs("") == []
    assert cb._parse_raw_pairs("   ") == []
