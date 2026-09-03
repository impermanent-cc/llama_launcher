from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.core.validation import validate


def _p():
    return Profile(
        name="p",
        image="img:tag",
        runtime=Runtime(node="box-b"),
        settings={"port": 8080},
    )


def test_missing_image_on_node_is_a_warning():
    issues = validate(_p(), binary_found=True, image_present=False)
    assert any(i.level == "warning" and "not present" in i.message for i in issues)


def test_present_image_produces_no_such_warning():
    issues = validate(_p(), binary_found=True, image_present=True)
    assert not any("not present" in i.message for i in issues)


def test_default_image_present_true_keeps_old_behavior():
    # no image_present kwarg -> defaults True -> no new warning
    issues = validate(_p(), binary_found=True)
    assert not any("not present" in i.message for i in issues)
