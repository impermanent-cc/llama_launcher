from llama_launcher.core.spec import Profile, Mount, Runtime
from llama_launcher.core.validation import validate, Issue


def _ok_profile():
    return Profile(
        name="p", image="img",
        mounts=[Mount(host="/h/models", container="/models", role="model", mode="ro")],
        model="/models/m.gguf", settings={"port": 8080},
    )


def test_valid_profile_has_no_errors():
    issues = validate(_ok_profile())
    assert [i for i in issues if i.level == "error"] == []


def test_missing_model_is_error():
    p = _ok_profile()
    p.model = ""
    errs = [i for i in validate(p) if i.level == "error"]
    assert any("model" in i.message.lower() for i in errs)


def test_model_not_under_mount_is_error():
    p = _ok_profile()
    p.model = "/elsewhere/m.gguf"
    errs = [i for i in validate(p) if i.level == "error"]
    assert any("mount" in i.message.lower() for i in errs)


def test_mmproj_and_lora_paths_checked():
    p = _ok_profile()
    p.mmproj = "/nope/x.gguf"
    errs = [i for i in validate(p) if i.level == "error"]
    assert any("mmproj" in i.message.lower() for i in errs)


def test_binary_not_found_is_error():
    errs = [i for i in validate(_ok_profile(), binary_found=False) if i.level == "error"]
    assert any("podman" in i.message.lower() or "runtime" in i.message.lower() for i in errs)


def test_tools_with_rw_model_mount_warns():
    p = _ok_profile()
    p.mounts[0].mode = "rw"
    p.settings["tools"] = "all"
    warns = [i for i in validate(p) if i.level == "warning"]
    assert any("writable" in i.message.lower() for i in warns)


def test_partial_mount_is_error():
    p = _ok_profile()
    p.mounts.append(Mount(host="/h/extra", container="", role="custom", mode="ro"))
    errs = [i for i in validate(p) if i.level == "error"]
    assert any("incomplete" in i.message.lower() for i in errs)


def test_fully_empty_mount_is_not_error():
    p = _ok_profile()
    p.mounts.append(Mount(host="", container="", role="custom", mode="ro"))
    errs = [i for i in validate(p) if i.level == "error"]
    assert not any("incomplete" in i.message.lower() for i in errs)


def test_port_collision_warns():
    p = _ok_profile()
    warns = [i for i in validate(p, running_ports=(8080,)) if i.level == "warning"]
    assert any("port" in i.message.lower() for i in warns)
