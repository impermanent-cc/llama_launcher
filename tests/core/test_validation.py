from llama_launcher.core.command_builder import raw_arg_warnings
from llama_launcher.core.spec import Profile, Mount, Runtime, RpcWorker
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


def test_port_choice_never_warns_about_discovery():
    # The old outside-harness discovery-scan warning is gone: any port is fine.
    for port in (8080, 9999):
        p = _router(settings={"port": port})
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


def test_blank_cors_origins_reads_as_upstream_wildcard():
    # A stored "" means --cors-origins is omitted, so the server runs with
    # upstream's '*' default: the exposed-bind wildcard warning must still
    # fire (older UI versions saved cors-origins as "" on untouched forms).
    from llama_launcher.core.spec import Runtime
    p = _router(runtime=Runtime(bind_host="0.0.0.0"),
                settings={"port": 8080, "cors-origins": ""})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert any("CORS origins '*'" in m for m in _warnings(issues))


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


from llama_launcher.core.spec import Profile, Runtime, Mount
from llama_launcher.core.validation import validate


def _ok_server(**kw):
    base = dict(
        name="p", image="ghcr.io/ikawrakow/ik-llama-cpp:cu12-server",
        runtime=Runtime(engine="ik_llama.cpp"),
        mounts=[Mount(host="/m", container="/models", role="model")],
        model="/models/x.gguf",
    )
    base.update(kw)
    return Profile(**base)


def _msgs(p):
    return [i.message for i in validate(p)]


def test_engine_ik_but_mainline_image_warns():
    p = _ok_server(image="ghcr.io/ggml-org/llama.cpp:server-cuda")
    assert any("doesn't look like an ik" in m for m in _msgs(p))
    assert all(i.level != "error" for i in validate(p) if "ik" in i.message)


def test_engine_llama_but_ik_image_warns():
    p = Profile(name="p", image="ghcr.io/ikawrakow/ik-llama-cpp:cu12-server",
                runtime=Runtime(engine="llama.cpp"),
                mounts=[Mount(host="/m", container="/models", role="model")],
                model="/models/x.gguf")
    assert any("looks like an ik_llama.cpp build" in m for m in _msgs(p))


def test_matched_engine_and_image_no_mismatch_warning():
    p = _ok_server()
    assert not any("ik" in m and "look" in m for m in _msgs(p))


def test_rtr_warns_and_mentions_mmap_override():
    p = _ok_server(settings={"run-time-repack": True})  # load-mode defaults to mmap
    msg = next(m for m in _msgs(p) if "run-time-repack" in m)
    assert "mmap" in msg


def test_no_rtr_no_warning():
    p = _ok_server(settings={})
    assert not any("run-time-repack" in m for m in _msgs(p))


def test_native_requires_existing_executable_binary():
    # core is I/O-free (tests/core/test_purity.py forbids `import os` here), so
    # the missing/non-executable filesystem check lives in
    # services.native.native_binary_available and is passed in as
    # `native_binary_ok`; a real stat of this path is covered separately in
    # tests/services/test_native.py.
    p = Profile(name="n", model="/m.gguf",
                runtime=Runtime(launch_mode="native", native_binary="/no/such/llama-server"))
    msgs = [i.message for i in validate(p, native_binary_ok=False)]
    assert any("binary" in m.lower() for m in msgs)


def test_native_binary_present_and_executable_passes():
    p = Profile(name="n", model="/m.gguf",
                runtime=Runtime(launch_mode="native", native_binary="/opt/bin/llama-server",
                                bind_host="127.0.0.1"))
    errors = [i for i in validate(p, native_binary_ok=True) if i.level == "error"]
    assert errors == []


def test_ik_engine_router_is_refused():
    """ik_llama.cpp has no router: its llama-server carries no --models-preset,
    so the container would die on "unknown argument" the moment it started.
    Verified by execution against ik-llama-cpp:cu12-server."""
    issues = validate(_router(runtime=Runtime(engine="ik_llama.cpp"), image="ik-llama-cpp"),
                      members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert any("router" in m.lower() and "ik_llama.cpp" in m for m in _errors(issues))


def test_router_on_an_ik_image_is_refused_even_with_the_mainline_engine():
    """The Runtime engine defaults to llama.cpp and nothing sets it from the
    image, so a router whose image is an ik build used to pass validate() with
    only the looks-ik warning, launch, and die on unknown argument
    --models-preset. Following that warning's advice then hit the engine
    refusal instead."""
    issues = validate(_router(image="ghcr.io/ikawrakow/ik-llama-cpp:cu12-server"),
                      members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert any("router" in m.lower() and "ik_llama.cpp" in m for m in _errors(issues))


def test_router_member_with_a_different_engine_is_refused():
    """The preset renders a member's settings for the member's engine, but the
    router's own (mainline) llama-server spawns every member, so an ik-tagged
    member writes ik-only keys into the preset and the child dies on unknown
    argument."""
    member = _member_profile(runtime=Runtime(engine="ik_llama.cpp"),
                             settings={"defer-experts": True})
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), member)],
                      api_key_present=True)
    errors = _errors(issues)
    assert any("Qwen" in m and "ik_llama.cpp" in m and "engine" in m.lower()
               for m in errors), errors


def test_router_member_with_the_same_engine_is_not_flagged():
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert not any("engine" in m.lower() for m in _errors(issues))


def test_mainline_engine_router_is_allowed():
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert not any("does not support router mode" in m for m in _errors(issues))


def test_native_router_is_refused():
    p = Profile(name="n", mode="router",
                runtime=Runtime(launch_mode="native", native_binary="/bin/sh"))
    msgs = [i.message.lower() for i in validate(p)]
    assert any("router" in m and "native" in m for m in msgs)


def test_native_profile_not_blocked_by_missing_container_runtime():
    # Runtime.binary defaults to "podman" for every profile regardless of
    # launch_mode; a native user (no podman installed) must not be blocked by
    # a check about a runtime the native path never uses.
    p = Profile(name="n", model="/m.gguf",
                runtime=Runtime(launch_mode="native", native_binary="/opt/bin/llama-server",
                                bind_host="127.0.0.1"))
    errors = [i for i in validate(p, binary_found=False, native_binary_ok=True)
              if i.level == "error"]
    assert errors == []


def test_container_profile_still_blocked_by_missing_runtime():
    p = _ok_profile()
    errs = [i for i in validate(p, binary_found=False) if i.level == "error"]
    assert any("not found on path" in m.message.lower() for m in errs)


def _rpc(workers, settings=None):
    return Profile(name="pool", image="img", model="/m/x.gguf",
                   mounts=[Mount(host="/m", container="/m")],
                   settings=settings or {},
                   runtime=Runtime(launch_mode="rpc", rpc_workers=workers))


def _levels(issues, needle):
    return [i.level for i in issues if needle in i.message]


def test_rpc_empty_pool_is_error():
    assert "error" in _levels(validate(_rpc([])), "at least one worker")


def test_rpc_missing_worker_image_is_error():
    issues = validate(_rpc([RpcWorker(node="box2")]),
                      worker_image_present={"box2": False})
    assert "error" in _levels(issues, "box2")


def test_rpc_overcommit_mem_is_warning():
    issues = validate(_rpc([RpcWorker(node="box2", mem_mb=64000)]),
                      worker_image_present={"box2": True},
                      worker_free_mb={"box2": 32000})
    assert "warning" in _levels(issues, "more than")


def test_rpc_cpu_moe_centralizing_warning():
    issues = validate(_rpc([RpcWorker(node="local")], settings={"cpu-moe": True}),
                      worker_image_present={"local": True})
    assert "warning" in _levels(issues, "centralizes")


def test_rpc_router_mode_is_error():
    # RPC pooling does not support router mode (GUI-only server-head pools).
    p = _rpc([RpcWorker(node="local")])
    p.mode = "router"
    issues = validate(p, worker_image_present={"local": True})
    assert "error" in _levels(issues, "does not support router")


def test_router_port_choice_raises_no_discovery_warning():
    """No outside-harness coupling: any router port is fine (the old rule
    warned when the port fell outside a specific tool's discovery scan)."""
    p = _router(settings={"port": 9123})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert not any("scan" in m.lower() or "odysseus" in m.lower()
                   for m in _warnings(issues) + _errors(issues))


def test_member_port_setting_warns_it_is_ignored():
    """llama.cpp spawns each member on its own random loopback port and strips
    --port from the preset, so a port set on a member profile silently does
    nothing; say so instead of letting it look like the router is randomizing."""
    member = _member_profile(settings={"port": 8090})
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), member)],
                      api_key_present=True)
    assert any("port" in m and "router" in m for m in _warnings(issues))


def test_member_without_port_setting_does_not_warn_about_ports():
    issues = validate(_router(), members=[(RouterMember(profile="Qwen"), _member_profile())],
                      api_key_present=True)
    assert not any("random" in m for m in _warnings(issues))


def test_router_ignores_leftover_single_server_warnings():
    """A router profile can carry the form's leftover draft model and member
    settings (kept so a Save in router mode is not destructive); the
    single-server launch warnings (inert draft, MTP limits) must not fire on
    a router launch -- the router itself loads no model."""
    p = _router(draft_model="/models/d.gguf",
                settings={"port": 8080, "spec-type": "draft-mtp",
                          "parallel": 4, "ctx-size": 8192})
    issues = validate(p, members=[(RouterMember(profile="Q"), _member_profile())],
                      api_key_present=True)
    assert not any("draft" in m.lower() or "mtp" in m.lower()
                   for m in _warnings(issues))


def test_server_inert_draft_warning_still_fires():
    p = Profile(name="S", image="img", model="/models/m.gguf",
                mounts=[Mount(host="/h", container="/models")],
                draft_model="/models/d.gguf", settings={"port": 8080})
    issues = validate(p)
    assert any("never used" in m for m in _warnings(issues))
