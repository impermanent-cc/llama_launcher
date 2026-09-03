import stat

import pytest

from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.services import api_key


def test_router_dir_is_slugified_and_created(tmp_path):
    d = api_key.router_dir(tmp_path, "My Router")
    assert d == tmp_path / "router" / "my-router"
    assert d.is_dir()


def test_generate_key_is_prefixed_and_unique():
    a, b = api_key.generate_key(), api_key.generate_key()
    assert a.startswith("sk-")
    assert len(a) > 20
    assert a != b


def test_ensure_api_key_creates_then_reuses(tmp_path):
    first = api_key.ensure_api_key(tmp_path, "R")
    second = api_key.ensure_api_key(tmp_path, "R")
    assert first == second


def test_key_file_is_owner_read_write_only(tmp_path):
    api_key.ensure_api_key(tmp_path, "R")
    path = api_key.router_dir(tmp_path, "R") / "api-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_read_api_key_returns_none_when_absent(tmp_path):
    assert api_key.read_api_key(tmp_path, "R") is None


def test_read_api_key_ignores_comments_and_blank_lines(tmp_path):
    d = api_key.router_dir(tmp_path, "R")
    (d / "api-key").write_text("# comment\n\nsk-real\n")
    assert api_key.read_api_key(tmp_path, "R") == "sk-real"


def test_write_preset_writes_into_the_router_dir(tmp_path):
    path = api_key.write_preset(tmp_path, "R", "version = 1\n")
    assert path == api_key.router_dir(tmp_path, "R") / "models.ini"
    assert path.read_text() == "version = 1\n"


def test_router_dir_is_not_group_writable(tmp_path):
    # The directory guards the key: if it is writable, another local user can
    # unlink api-key and choose the router's credential, which the 0600 on the
    # file does nothing to prevent.
    d = api_key.router_dir(tmp_path, "R")
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_reading_a_key_does_not_create_directories(tmp_path):
    # read_api_key runs on the monitor tick; a poll must not touch the disk.
    assert api_key.read_api_key(tmp_path, "Never Launched") is None
    assert not (tmp_path / "router" / "never-launched").exists()


def test_key_file_is_never_world_readable_even_briefly(tmp_path, monkeypatch):
    # Guards the write-then-chmod window: with chmod neutered, the file must
    # still have been created 0600 by open(2) itself.
    monkeypatch.setattr("pathlib.Path.chmod", lambda self, mode: None)
    api_key.ensure_api_key(tmp_path, "R")
    path = api_key.router_dir(tmp_path, "R") / "api-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_normalize_strips_surrounding_whitespace():
    assert api_key.normalize_key("  sk-abc \n") == "sk-abc"


@pytest.mark.parametrize("bad", ["", "   ", "a\nb"])
def test_normalize_rejects_empty_or_multiline(bad):
    with pytest.raises(ValueError):
        api_key.normalize_key(bad)


def test_write_and_read_global_key(tmp_path):
    assert api_key.read_global_key(tmp_path) is None
    api_key.write_global_key(tmp_path, "sk-global")
    assert api_key.read_global_key(tmp_path) == "sk-global"


def test_global_key_file_is_0600(tmp_path):
    api_key.write_global_key(tmp_path, "sk-global")
    mode = api_key.global_key_path(tmp_path).stat().st_mode & 0o777
    assert mode == 0o600


def test_write_global_key_tightens_a_preexisting_wider_router_dir(tmp_path):
    # router_dir() as an intermediate parent (e.g. a per-profile key written
    # first) creates base_dir/router 0700 via mkdir -p, but exist_ok=True on a
    # later mkdir call would not re-tighten it if something else widened it.
    d = api_key.router_dir(tmp_path, "R")  # creates base_dir/router along the way
    (tmp_path / "router").chmod(0o755)
    api_key.write_global_key(tmp_path, "sk-global")
    mode = stat.S_IMODE((tmp_path / "router").stat().st_mode)
    assert mode == 0o700
    assert d.exists()  # sanity: didn't touch the per-router subdirectory


def test_set_profile_key_overwrites_and_reads_back(tmp_path):
    api_key.set_profile_key(tmp_path, "R", "sk-mine")
    assert api_key.read_api_key(tmp_path, "R") == "sk-mine"
    mode = api_key._key_path(tmp_path, "R", create=False).stat().st_mode & 0o777
    assert mode == 0o600


# -- resolver + launch-time materializer -------------------------------------
def _profile(name, mode):
    return Profile(name=name, image="img", runtime=Runtime(router_key_mode=mode))


def test_resolve_global_prefers_global_key(tmp_path):
    api_key.ensure_api_key(tmp_path, "R")  # a per-profile key exists
    api_key.write_global_key(tmp_path, "sk-shared")
    assert api_key.resolve_api_key(tmp_path, _profile("R", "global")) == "sk-shared"


def test_resolve_global_falls_back_to_per_profile_when_no_global(tmp_path):
    legacy = api_key.ensure_api_key(tmp_path, "R")
    assert api_key.resolve_api_key(tmp_path, _profile("R", "global")) == legacy


def test_resolve_own_ignores_global(tmp_path):
    api_key.set_profile_key(tmp_path, "R", "sk-own")
    api_key.write_global_key(tmp_path, "sk-shared")
    assert api_key.resolve_api_key(tmp_path, _profile("R", "own")) == "sk-own"


def test_resolve_returns_none_when_nothing_set(tmp_path):
    assert api_key.resolve_api_key(tmp_path, _profile("R", "global")) is None


def test_resolve_is_side_effect_free(tmp_path):
    # global mode, no keys -> must NOT create a per-profile file
    api_key.resolve_api_key(tmp_path, _profile("R", "global"))
    assert api_key.read_api_key(tmp_path, "R") is None


def test_prepare_launch_materializes_global_into_per_profile_file(tmp_path):
    api_key.write_global_key(tmp_path, "sk-shared")
    key = api_key.prepare_launch_key(tmp_path, _profile("R", "global"))
    assert key == "sk-shared"
    # the container reads the per-profile file, so it must now hold the global key
    assert api_key.read_api_key(tmp_path, "R") == "sk-shared"


def test_prepare_launch_generates_when_no_global(tmp_path):
    key = api_key.prepare_launch_key(tmp_path, _profile("R", "global"))
    assert key and api_key.read_api_key(tmp_path, "R") == key


def test_prepare_launch_own_uses_per_profile(tmp_path):
    api_key.set_profile_key(tmp_path, "R", "sk-own")
    api_key.write_global_key(tmp_path, "sk-shared")
    assert api_key.prepare_launch_key(tmp_path, _profile("R", "own")) == "sk-own"
