from llama_launcher.core.command_builder import raw_arg_warnings
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


def _mtp_profile():
    p = _ok_profile()
    p.settings["spec-type"] = "draft-mtp"
    return p


def test_mtp_with_mmproj_warns():
    p = _mtp_profile()
    p.mmproj = "/models/mmproj.gguf"   # under the mount, so no path error
    warns = [i for i in validate(p) if i.level == "warning"]
    assert any("mtp" in i.message.lower() and "mmproj" in i.message.lower() for i in warns)


def test_mtp_with_parallel_gt1_warns():
    p = _mtp_profile()
    p.settings["parallel"] = 4
    warns = [i for i in validate(p) if i.level == "warning"]
    assert any("mtp" in i.message.lower() and "parallel" in i.message.lower() for i in warns)


def test_mtp_single_slot_text_only_has_no_mtp_warning():
    p = _mtp_profile()
    p.settings["parallel"] = 1   # mmproj unset, single slot -> nothing to warn about
    assert not any("mtp" in i.message.lower() for i in validate(p))


def test_non_mtp_profile_gets_no_mtp_warning():
    p = _ok_profile()                       # no spec-type
    p.mmproj = "/models/mmproj.gguf"
    p.settings["parallel"] = 4
    assert not any("mtp" in i.message.lower() for i in validate(p))


def test_draft_model_without_spec_type_warns():
    p = _ok_profile()
    p.draft_model = "/models/draft.gguf"    # loaded but spec-type left at 'none'
    warns = [i for i in validate(p) if i.level == "warning"]
    assert any("draft" in i.message.lower() and "spec-type" in i.message.lower()
               for i in warns)


def test_draft_model_with_spec_type_has_no_draft_warning():
    p = _ok_profile()
    p.draft_model = "/models/draft.gguf"
    p.settings["spec-type"] = "draft-simple"   # now actually used
    assert not any("draft model" in i.message.lower() for i in validate(p))


def test_no_draft_model_has_no_draft_warning():
    p = _ok_profile()
    assert not any("draft model" in i.message.lower() for i in validate(p))


def _vprofile(**settings):
    from llama_launcher.core.spec import Profile, Mount, Runtime
    return Profile(
        name="e", image="img",
        runtime=Runtime(binary="podman", gpu_mode="cdi"),
        mounts=[Mount(host="/m", container="/models", role="model", mode="ro")],
        model="/models/x.gguf", settings={**settings},
    )


def test_reranking_without_rank_pooling_warns():
    issues = validate(_vprofile(reranking=True, embeddings=True, pooling="mean"))
    assert any(i.level == "warning" and "rank" in i.message for i in issues)


def test_reranking_without_embeddings_warns():
    issues = validate(_vprofile(reranking=True, pooling="rank"))
    assert any(i.level == "warning" and "embedding" in i.message.lower() for i in issues)


def test_sampling_changed_in_embedding_mode_warns():
    issues = validate(_vprofile(embeddings=True, temp=0.5))
    assert any("sampling" in i.message.lower() for i in issues)


def test_clean_embedding_profile_has_no_embed_warnings():
    issues = validate(_vprofile(embeddings=True, pooling="mean"))
    assert not any("rank" in i.message or "sampling" in i.message.lower() for i in issues)


from llama_launcher.core.spec import Mount, Profile, RouterMember, Runtime
from llama_launcher.core.validation import validate


def _router(**kw):
    base = dict(
        name="Host", mode="router", image="img",
        mounts=[Mount(host="/mnt/models", container="/models")],
        members=[RouterMember(profile="Qwen")],
        settings={"port": 8080},
    )
    base.update(kw)
    return Profile(**base)


def _member_profile(name="Qwen", model="/models/qwen.gguf", **kw):
    return Profile(name=name, model=model, **kw)


def _errors(issues):
    return [i.message for i in issues if i.level == "error"]


def _warnings(issues):
    return [i.message for i in issues if i.level == "warning"]


def test_router_does_not_require_its_own_model():
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert not any("No model selected" in m for m in _errors(issues))


def test_router_with_no_members_is_an_error():
    issues = validate(_router(members=[]), members=[], api_key_present=True)
    assert any("at least one model" in m for m in _errors(issues))


def test_duplicate_model_ids_are_an_error():
    m1 = RouterMember(profile="Qwen Big")
    m2 = RouterMember(profile="Qwen  Big")       # slugifies to the same id
    issues = validate(_router(), api_key_present=True,
                      members=[(m1, _member_profile("Qwen Big")),
                               (m2, _member_profile("Qwen  Big"))])
    assert any("qwen-big" in m for m in _errors(issues))


def test_member_model_must_be_under_a_router_mount():
    member = (RouterMember(profile="Q"), _member_profile(model="/elsewhere/q.gguf"))
    issues = validate(_router(), members=[member], api_key_present=True)
    assert any("not under any mount" in m for m in _errors(issues))


def test_non_loopback_bind_without_api_key_is_an_error():
    p = _router(runtime=Runtime(bind_host="0.0.0.0"))
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=False)
    assert any("without an API key" in m for m in _errors(issues))


def test_non_loopback_bind_with_api_key_is_allowed():
    p = _router(runtime=Runtime(bind_host="0.0.0.0"))
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert not any("without an API key" in m for m in _errors(issues))


def test_models_max_above_one_warns():
    p = _router(settings={"port": 8080, "models-max": 3})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert any("models-max" in m for m in _warnings(issues))


def test_port_outside_discovery_ranges_warns():
    p = _router(settings={"port": 9999})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert any("discover" in m.lower() for m in _warnings(issues))


def test_port_inside_discovery_ranges_does_not_warn():
    p = _router(settings={"port": 8080})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert not any("discover" in m.lower() for m in _warnings(issues))


def test_multi_lora_member_warns():
    from llama_launcher.core.spec import LoraRef
    member_profile = _member_profile(loras=[LoraRef(path="/models/a.gguf"),
                                            LoraRef(path="/models/b.gguf")])
    issues = validate(_router(), api_key_present=True,
                      members=[(RouterMember(profile="Q"), member_profile)])
    assert any("LoRA" in m for m in _warnings(issues))


def test_unconvertible_member_raw_args_warns():
    member_profile = _member_profile(raw_args="--foo a --foo b")
    issues = validate(_router(), api_key_present=True,
                      members=[(RouterMember(profile="Q"), member_profile)])
    assert any("preset" in m.lower() for m in _warnings(issues))


def test_tools_with_lan_cors_origin_warns():
    p = _router(settings={"port": 8080, "tools": "all",
                          "cors-origins": "http://vm.local"})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert any("clamp" in m.lower() for m in _warnings(issues))


def test_server_mode_validation_unchanged():
    p = Profile(name="Solo", image="img")
    assert any("No model selected" in m for m in _errors(validate(p)))


def test_server_mode_non_loopback_bind_without_key_is_an_error():
    # The Bind address control is shown in BOTH modes, so the exposure guard
    # cannot live only in the router branch.
    p = Profile(name="Solo", image="img", model="/models/a.gguf",
                mounts=[Mount(host="/h", container="/models")],
                runtime=Runtime(bind_host="0.0.0.0"), settings={"port": 8080})
    assert any("without an API key" in m for m in _errors(validate(p)))


def test_server_mode_non_loopback_bind_with_a_typed_key_is_allowed():
    p = Profile(name="Solo", image="img", model="/models/a.gguf",
                mounts=[Mount(host="/h", container="/models")],
                runtime=Runtime(bind_host="0.0.0.0"),
                settings={"port": 8080, "api-key": "sk-typed"})
    assert not any("without an API key" in m for m in _errors(validate(p)))


def test_server_mode_loopback_bind_is_silent():
    p = Profile(name="Solo", image="img", model="/models/a.gguf",
                mounts=[Mount(host="/h", container="/models")],
                settings={"port": 8080})
    assert not any("without an API key" in m for m in _errors(validate(p)))


def _srv(raw="", **settings):
    return Profile(
        name="s", image="img", runtime=Runtime(bind_host="127.0.0.1"), mode="server",
        mounts=[Mount(host="/h", container="/models", role="model")],
        model="/models/m.gguf", raw_args=raw, settings={"port": 8080, **settings},
    )


def test_raw_arg_warnings_reports_override_and_protected():
    warns = raw_arg_warnings(_srv(raw="-ngl 50 --port 9000", **{"n-gpu-layers": "99"}))
    assert any("overrides '--n-gpu-layers'" in w for w in warns)
    assert any("--port" in w and "ignored" in w for w in warns)


def test_raw_arg_warnings_empty_when_no_collision():
    assert raw_arg_warnings(_srv(raw="--numa distribute")) == []


def test_validate_emits_warning_issue_for_raw_collision():
    issues = validate(_srv(raw="-ngl 50", **{"n-gpu-layers": "99"}))
    assert any(i.level == "warning" and "overrides '--n-gpu-layers'" in i.message
               for i in issues)


def test_validate_no_raw_warning_when_clean():
    issues = validate(_srv(raw="--numa distribute"))
    assert not any("overrides" in i.message for i in issues)
