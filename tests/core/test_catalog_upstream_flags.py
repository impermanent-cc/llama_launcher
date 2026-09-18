"""Every catalog flag must be one the engine it reaches actually accepts.

This guards a failure mode the rest of the suite cannot see: the other catalog
tests assert that the app EMITS a flag, which stays green even when the server
rejects it (a flag upstream has renamed, or a mainline-only flag reaching an
ik_llama.cpp launch).

Three engines, three fixtures, and they are NOT gathered the same way:

* mainline (`llama_server_flags_b10902.txt`) is a straight `--help` dump.
* ik (`ik_llama_server_flags_cu12.txt`) is `--help` PLUS flags confirmed by
  executing them, because ik's help under-reports its own parser.
* PrismML (`prism_llama_server_flags_b10685.txt`) is a straight `--help` dump
  of the release tarball's binary, captured by regen_prism_flags.sh.

Regenerate with tests/fixtures/regen_flags.sh, regen_ik_flags.sh and
regen_prism_flags.sh when moving to newer images, then read the diff: a flag
that disappears means the engine renamed or dropped it, and the catalog has
to follow.
"""

import functools
import pathlib

import pytest

from llama_launcher.core.command_builder import _STRUCTURAL_ALIASES
from llama_launcher.core.settings_catalog import (
    CATALOG,
    MAINLINE_ONLY_FLAGS,
    PRISM_LACKS_FLAGS,
    engines_of,
    for_engine,
)

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "llama_server_flags_b10902.txt"
)

# The parametrised cases below select with for_engine(), the exact predicate
# that decides what reaches each engine's launch, so the tests cannot drift
# from what the launcher emits.
MAINLINE_KEYS = sorted(for_engine(CATALOG, "llama.cpp"))
IK_KEYS = sorted(for_engine(CATALOG, "ik_llama.cpp"))
PRISM_KEYS = sorted(for_engine(CATALOG, "prism_llama.cpp"))


def _flags_in(path: pathlib.Path) -> frozenset[str]:
    return frozenset(
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )


@functools.cache
def _upstream_flags() -> frozenset[str]:
    return _flags_in(FIXTURE)


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
    return _flags_in(IK_FIXTURE)


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
    assert any(CATALOG[k].engine == "prism_llama.cpp" for k in PRISM_KEYS)
    assert not any(CATALOG[k].engine == "llama.cpp" for k in PRISM_KEYS)
    assert not any(CATALOG[k].engine == "ik_llama.cpp" for k in PRISM_KEYS)


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
    # dropped, every test above would still pass while ik launches broke. Settings
    # gated away from ik_llama.cpp by an explicit engine= on their own Setting()
    # call (kv-mean-center, the legacy load pair) are excluded: they were never
    # "any" before the pass and so are not evidence the pass ran.
    from llama_launcher.core.spec import MAINLINE_FAMILY

    tagged = {
        s.flag
        for s in CATALOG.values()
        if s.engine in ("llama.cpp", MAINLINE_FAMILY)
        and "ik_llama.cpp" not in engines_of(s)
    }
    assert tagged == set(MAINLINE_ONLY_FLAGS)


UNEXPOSED_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "unexposed_flags_mainline.txt"
)


@functools.cache
def _unexposed_flags() -> frozenset[str]:
    return _flags_in(UNEXPOSED_FIXTURE)


def _catalog_covered_flags() -> frozenset[str]:
    # Aliases count: a flag the catalog reaches under any spelling is
    # accounted for, even though only the primary flag is emitted.
    out = set()
    for s in for_engine(CATALOG, "llama.cpp").values():
        out.add(s.flag)
        out.update(s.aliases or ())
    # These reach argv directly from _owned_server_pairs rather than through
    # a catalogued Setting, so they are exposed even though no Setting names
    # them.
    out.update(_STRUCTURAL_ALIASES)
    return frozenset(out)


def test_every_accepted_flag_is_catalogued_or_recorded_as_unexposed():
    """The audit's other direction: a flag the build accepts that nothing
    accounts for is a new upstream flag nobody has decided about."""
    unaccounted = sorted(
        _upstream_flags() - _catalog_covered_flags() - _unexposed_flags()
    )
    assert unaccounted == [], (
        f"llama-server accepts these and the catalog neither exposes nor "
        f"records them: {unaccounted}. Add a Setting, register it in "
        f"_STRUCTURAL_ALIASES if the launcher emits it directly, or add each "
        f"to {UNEXPOSED_FIXTURE.name} to record it as deliberately unexposed."
    )


def test_unexposed_list_carries_nothing_the_build_rejects():
    """A stale entry hides a removal: if upstream drops a flag, the line here
    has to go too."""
    stale = sorted(_unexposed_flags() - _upstream_flags())
    assert stale == [], (
        f"{UNEXPOSED_FIXTURE.name} lists flags this build does not accept: {stale}"
    )


def test_unexposed_list_and_the_catalog_do_not_overlap():
    """A flag cannot be both exposed and deliberately unexposed; an overlap
    means a Setting was added and the list not trimmed."""
    both = sorted(_unexposed_flags() & _catalog_covered_flags())
    assert both == [], f"listed as unexposed but catalogued: {both}"


def test_structural_aliases_are_accepted_by_the_build():
    """The launcher emits or folds these directly rather than through a
    Setting, so a rename upstream breaks every launch while the catalogue
    tests stay green."""
    missing = sorted(f for f in _STRUCTURAL_ALIASES if f not in _upstream_flags())
    assert missing == [], (
        f"llama-server does not accept {missing}, which the launcher emits "
        "or folds directly. Update the spelling where it is emitted or "
        "folded, in command_builder._owned_server_pairs, "
        "command_builder._STRUCTURAL_ALIASES, or services.sweep.sweep_profile, "
        "not the fixture."
    )


# -- PrismML llama.cpp --------------------------------------------------------
# The fork's server is mainline's parser at build 10685: it lacks the flags
# mainline added after that build and keeps the pair mainline dropped at 10902.

PRISM_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "prism_llama_server_flags_b10685.txt"
)
PRISM_UNEXPOSED_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "unexposed_flags_prism.txt"
)
# The sweep appends --no-log-jsonl beside a raw --log-jsonl alone, and build
# 10685 knows neither, so this is the one structural flag the fork never sees.
PRISM_STRUCTURAL_EXEMPT = frozenset({"--no-log-jsonl"})


@functools.cache
def _prism_flags() -> frozenset[str]:
    return _flags_in(PRISM_FIXTURE)


@functools.cache
def _prism_unexposed() -> frozenset[str]:
    return _flags_in(PRISM_UNEXPOSED_FIXTURE)


def _prism_covered_flags() -> frozenset[str]:
    out = set()
    for s in for_engine(CATALOG, "prism_llama.cpp").values():
        out.add(s.flag)
        out.update(s.aliases or ())
    out.update(_STRUCTURAL_ALIASES)
    return frozenset(out)


def test_prism_fixture_looks_like_a_real_help_dump():
    flags = _prism_flags()
    assert len(flags) > 200
    for anchor in (
        "--port",
        "--host",
        "--model",
        "--lora",
        "--ctx-size",
        "--load-mode",
        "--mlock",
        "--kv-mean-center",
    ):
        assert anchor in flags


@pytest.mark.parametrize("key", PRISM_KEYS)
def test_setting_reaching_prism_is_accepted_by_prism(key):
    setting = CATALOG[key]
    assert setting.flag in _prism_flags(), (
        f"{key}: engine={setting.engine!r} means this reaches a PrismML launch, "
        f"but the release does not accept {setting.flag}. If mainline added it "
        "after build 10685, add it to PRISM_LACKS_FLAGS; if it is PrismML-only, "
        "fix the spelling or drop the setting."
    )


def test_prism_lacks_flags_are_absent_from_prism_and_present_on_mainline():
    """Both directions of the lacking list: a listed flag the release does
    accept hides a working setting, and a listed flag mainline no longer
    accepts is dead weight."""
    wrongly_hidden = sorted(f for f in PRISM_LACKS_FLAGS if f in _prism_flags())
    assert wrongly_hidden == []
    gone_upstream = sorted(f for f in PRISM_LACKS_FLAGS if f not in _upstream_flags())
    assert gone_upstream == []


def test_every_mainline_flag_prism_rejects_is_listed_as_lacking():
    """Exactly the lacking list separates the two mainline-family catalogs."""
    rejected = sorted(
        {
            s.flag
            for s in for_engine(CATALOG, "llama.cpp").values()
            if s.flag not in _prism_flags()
        }
    )
    assert rejected == sorted(PRISM_LACKS_FLAGS)


def test_every_flag_prism_accepts_is_catalogued_or_recorded_as_unexposed():
    unaccounted = sorted(
        _prism_flags()
        - _prism_covered_flags()
        - _unexposed_flags()
        - _prism_unexposed()
    )
    assert unaccounted == [], (
        f"PrismML llama-server accepts these and nothing accounts for them: "
        f"{unaccounted}. Add a Setting, or add each to "
        f"{PRISM_UNEXPOSED_FIXTURE.name} (or the mainline list if mainline "
        "accepts it too) to record it as deliberately unexposed."
    )


def test_prism_unexposed_list_carries_nothing_the_release_rejects():
    stale = sorted(_prism_unexposed() - _prism_flags())
    assert stale == [], f"{PRISM_UNEXPOSED_FIXTURE.name} lists rejected flags: {stale}"


def test_prism_unexposed_list_repeats_nothing_and_overlaps_no_setting():
    repeated = sorted(_prism_unexposed() & _unexposed_flags())
    assert repeated == [], f"already in the mainline list: {repeated}"
    both = sorted(_prism_unexposed() & _prism_covered_flags())
    assert both == [], f"listed as unexposed but catalogued: {both}"


def test_structural_aliases_are_accepted_by_prism():
    missing = sorted(
        f
        for f in _STRUCTURAL_ALIASES
        if f not in _prism_flags() and f not in PRISM_STRUCTURAL_EXEMPT
    )
    assert missing == []
    assert PRISM_STRUCTURAL_EXEMPT <= set(_STRUCTURAL_ALIASES)
    assert not (PRISM_STRUCTURAL_EXEMPT & _prism_flags())
