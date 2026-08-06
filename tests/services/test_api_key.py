import stat

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


def test_regenerate_api_key_replaces_the_old_one(tmp_path):
    first = api_key.ensure_api_key(tmp_path, "R")
    second = api_key.regenerate_api_key(tmp_path, "R")
    assert second != first
    assert api_key.read_api_key(tmp_path, "R") == second


def test_write_preset_writes_into_the_router_dir(tmp_path):
    path = api_key.write_preset(tmp_path, "R", "version = 1\n")
    assert path == api_key.router_dir(tmp_path, "R") / "models.ini"
    assert path.read_text() == "version = 1\n"
