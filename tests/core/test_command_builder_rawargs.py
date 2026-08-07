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
