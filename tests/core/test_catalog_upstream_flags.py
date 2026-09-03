"""Every catalog flag must be one the engine it reaches actually accepts.

This guards a failure mode the rest of the suite cannot see: the other catalog
tests assert that the app EMITS a flag, which stays green even when the server
rejects it (a flag upstream has renamed, or a mainline-only flag reaching an
ik_llama.cpp launch).

Two engines, two fixtures, and they are NOT gathered the same way:

* mainline (`llama_server_flags_b10711.txt`) is a straight `--help` dump.
* ik (`ik_llama_server_flags_cu12.txt`) is `--help` PLUS flags confirmed by
  executing them, because ik's help under-reports its own parser.

Regenerate with tests/fixtures/regen_flags.sh and regen_ik_flags.sh when moving
to newer images, then read the diff: a flag that disappears means the engine
renamed or dropped it, and the catalog has to follow.
"""

import functools
import pathlib

import pytest

from llama_launcher.core.settings_catalog import (
    CATALOG,
    MAINLINE_ONLY_FLAGS,
    for_engine,
)

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "llama_server_flags_b10711.txt"
)

# The parametrised cases below select with for_engine(), the exact predicate
# that decides what reaches each engine's launch, so the tests cannot drift
# from what the launcher emits.
MAINLINE_KEYS = sorted(for_engine(CATALOG, "llama.cpp"))
IK_KEYS = sorted(for_engine(CATALOG, "ik_llama.cpp"))


@functools.cache
def _upstream_flags() -> frozenset[str]:
    return frozenset(
        ln.strip()
        for ln in FIXTURE.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )


def test_fixture_looks_like_a_real_help_dump():
    flags = _upstream_flags()
    # Cheap sanity check so a truncated or empty fixture fails loudly here
    # rather than silently passing every case below.
    assert len(flags) > 200
    for anchor in ("--port", "--host", "--model", "--lora", "--ctx-size"):
        assert anchor in flags


@pytest.mark.parametrize("key", MAINLINE_KEYS)
def test_mainline_setting_uses_a_flag_upstream_accepts(key):
    # Asserts on setting.flag alone, NOT on the aliases: _render_setting always
    # emits the primary flag, so a setting whose flag was renamed still breaks
    # the launch even when one of its aliases survives upstream. Accepting an
    # alias here would let exactly that through.
    setting = CATALOG[key]
    assert setting.flag in _upstream_flags(), (
        f"{key}: llama-server does not accept {setting.flag} (aliases "
        f"{sorted(setting.aliases)} are not emitted, so they cannot save it). "
        "If upstream renamed it, point the Setting at the new spelling and keep "
        "the key so saved profiles survive."
    )


def test_engine_gated_settings_are_not_checked_against_mainline():
    # ik_llama.cpp flags are absent from mainline help by design; the parametrised
    # test above must not be quietly checking (and passing) them.
    ik = [k for k, s in CATALOG.items() if s.engine == "ik_llama.cpp"]
    assert ik, "expected some ik_llama.cpp-gated settings"
    upstream = _upstream_flags()
    assert any(CATALOG[k].flag not in upstream for k in ik)


# -- ik_llama.cpp -------------------------------------------------------------
# ik is a fork and rejects a large slice of the shared flag surface. A setting
# left at engine="any" that ik does not accept breaks an ik launch the moment a
# user sets it, and so does a typo or a future ik rename in an ik-only flag, so
# both the "any" and the "ik_llama.cpp" buckets are under test here.

IK_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "ik_llama_server_flags_cu12.txt"
)


@functools.cache
def _ik_flags() -> frozenset[str]:
    return frozenset(
        ln.strip()
        for ln in IK_FIXTURE.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )


def test_ik_fixture_looks_like_a_real_capture():
    flags = _ik_flags()
    assert len(flags) > 200
    for anchor in ("--port", "--host", "--model", "--ctx-size"):
        assert anchor in flags
    # Captured from a CUDA build on purpose: a CPU-only ik image omits the GPU
    # flags, which would wrongly mark them unsupported and retag them away.
    assert "--tensor-split" in flags
    # And supplemented by execution, since ik's --help omits these.
    assert "--n-gpu-layers" in flags


@pytest.mark.parametrize("key", IK_KEYS)
def test_setting_reaching_ik_is_accepted_by_ik(key):
    setting = CATALOG[key]
    if setting.engine == "any":
        hint = (
            "Add it to MAINLINE_ONLY_FLAGS so it is retagged engine='llama.cpp' "
            "and dropped from ik launches and forms."
        )
    else:
        hint = "It is ik-only, so fix the spelling or drop the setting."
    assert setting.flag in _ik_flags(), (
        f"{key}: engine={setting.engine!r} means this reaches an ik_llama.cpp "
        f"launch, but ik does not accept {setting.flag}. {hint}"
    )


def test_parametrised_selections_cover_the_engine_specific_buckets():
    # The ik selection must include the engine="ik_llama.cpp" bucket, not only
    # engine="any"; otherwise a typo in an ik-only flag keeps the suite green
    # while the ik launch dies. Guard the selections themselves, not the
    # per-key assertion.
    assert any(CATALOG[k].engine == "ik_llama.cpp" for k in IK_KEYS)
    assert any(CATALOG[k].engine == "llama.cpp" for k in MAINLINE_KEYS)
    assert not any(CATALOG[k].engine == "llama.cpp" for k in IK_KEYS)
    assert not any(CATALOG[k].engine == "ik_llama.cpp" for k in MAINLINE_KEYS)


def test_mainline_only_flags_are_actually_absent_from_ik():
    # Guards the opposite mistake: retagging something ik DOES accept, which
    # would silently remove a working setting from every ik profile's form.
    ik = _ik_flags()
    wrongly_excluded = sorted(f for f in MAINLINE_ONLY_FLAGS if f in ik)
    assert wrongly_excluded == [], (
        f"these are accepted by ik and must not be in MAINLINE_ONLY_FLAGS: "
        f"{wrongly_excluded}"
    )


def test_every_mainline_only_flag_still_exists_in_the_catalog():
    # A stale entry here is dead weight that hides a real rename; if a flag is
    # renamed, the MAINLINE_ONLY_FLAGS entry has to follow it.
    catalog_flags = {s.flag for s in CATALOG.values()}
    stale = sorted(f for f in MAINLINE_ONLY_FLAGS if f not in catalog_flags)
    assert stale == [], f"MAINLINE_ONLY_FLAGS references unknown flags: {stale}"


def test_retag_actually_took_effect():
    # MAINLINE_ONLY_FLAGS is applied by a post-pass over _ALL; if that pass were
    # dropped, every test above would still pass while ik launches broke.
    tagged = {s.flag for s in CATALOG.values() if s.engine == "llama.cpp"}
    assert tagged == set(MAINLINE_ONLY_FLAGS)
