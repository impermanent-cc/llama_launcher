from llama_launcher.core.build_spec import BuildConfig, BuildOutput
from llama_launcher.store.builds import (
    save_build_config, list_build_configs, delete_build_config,
    load_outputs, add_output, remove_output, write_containerfile,
    new_output_id,
)


def _out(oid="a1", ident="llama-custom:x-20260828"):
    return BuildOutput(id=oid, kind="tag", identifier=ident, config_name="x",
                       engine="llama.cpp", git_ref="master", options={},
                       created="2026-08-28")


def test_config_save_list_delete(tmp_path):
    save_build_config(BuildConfig(name="Cuda Perf"), tmp_path)
    assert [c.name for c in list_build_configs(tmp_path)] == ["Cuda Perf"]
    delete_build_config("Cuda Perf", tmp_path)
    assert list_build_configs(tmp_path) == []


def test_corrupt_config_skipped(tmp_path):
    save_build_config(BuildConfig(name="ok"), tmp_path)
    (tmp_path / "builds" / "bad.json").write_text("{nope")
    assert [c.name for c in list_build_configs(tmp_path)] == ["ok"]


def test_outputs_roundtrip_and_remove(tmp_path):
    add_output(_out("a1"), tmp_path)
    add_output(_out("b2", "ik-custom:y-20260828"), tmp_path)
    assert {o.id for o in load_outputs(tmp_path)} == {"a1", "b2"}
    remove_output("a1", tmp_path)
    assert [o.id for o in load_outputs(tmp_path)] == ["b2"]


def test_corrupt_outputs_recovers_empty(tmp_path):
    add_output(_out(), tmp_path)
    (tmp_path / "builds" / "outputs.json").write_text("[{")
    assert load_outputs(tmp_path) == []
    assert (tmp_path / "builds" / "outputs.json.bad").exists()


def test_write_containerfile(tmp_path):
    p = write_containerfile(BuildConfig(name="srv"), "FROM x\n", tmp_path)
    assert p.name == "srv.containerfile"
    assert p.read_text() == "FROM x\n"


def test_new_output_id_shape():
    a, b = new_output_id(), new_output_id()
    assert a != b and len(a) == 12
