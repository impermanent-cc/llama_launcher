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


PORTCANON = {"--host", "--port"}
LORACANON = {"--lora", "--lora-scaled"}


def test_merge_override_raw_wins_in_place():
    owned = [("-m", "/m.gguf"), ("--n-gpu-layers", "99"), ("--host", "0.0.0.0")]
    argv, warns = cb._merge_raw_args(owned, [("-ngl", "50")], PORTCANON, LORACANON)
    assert argv == ["-m", "/m.gguf", "--n-gpu-layers", "50", "--host", "0.0.0.0"]
    assert len(warns) == 1 and "overrides '--n-gpu-layers'" in warns[0] and "was 99" in warns[0]


def test_merge_alias_folds_both_directions():
    # raw long form overrides owned short form
    owned = [("-ngl", "99")]
    argv, warns = cb._merge_raw_args(owned, [("--n-gpu-layers", "7")], PORTCANON, LORACANON)
    assert argv == ["-ngl", "7"] and len(warns) == 1


def test_merge_protected_port_dropped_and_warned():
    owned = [("--port", "8080")]
    argv, warns = cb._merge_raw_args(owned, [("--port", "9000")], PORTCANON, LORACANON)
    assert argv == ["--port", "8080"]
    assert len(warns) == 1 and "ignored" in warns[0] and "--port" in warns[0]


def test_merge_repeatable_lora_appends_no_warning():
    owned = [("--lora", "/b.gguf")]
    argv, warns = cb._merge_raw_args(owned, [("--lora", "/a.gguf")], PORTCANON, LORACANON)
    assert argv == ["--lora", "/b.gguf", "--lora", "/a.gguf"]
    assert warns == []


def test_merge_unknown_flag_appended_no_warning():
    owned = [("--ctx-size", "4096")]
    argv, warns = cb._merge_raw_args(owned, [("--numa", "distribute")], PORTCANON, LORACANON)
    assert argv == ["--ctx-size", "4096", "--numa", "distribute"]
    assert warns == []


def test_merge_bare_bool_duplicate_warns():
    owned = [("--mlock", None)]
    argv, warns = cb._merge_raw_args(owned, [("--mlock", None)], PORTCANON, LORACANON)
    assert argv == ["--mlock"]
    assert len(warns) == 1 and "duplicates" in warns[0]


def test_merge_no_raw_is_owned_unchanged():
    owned = [("-m", "/m.gguf"), ("--host", "0.0.0.0"), ("--port", "8080")]
    argv, warns = cb._merge_raw_args(owned, [], PORTCANON, LORACANON)
    assert argv == ["-m", "/m.gguf", "--host", "0.0.0.0", "--port", "8080"]
    assert warns == []
