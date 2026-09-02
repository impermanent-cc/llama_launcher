"""Trust-boundary hardening: a shared/loaded profile.json must not be able to
escalate to host code execution or silently expose an unauthenticated server
on one Launch click. validate() screens the dangerous constructs.
"""
from llama_launcher.core.spec import Profile, Mount, Runtime, RouterMember
from llama_launcher.core.validation import validate
from llama_launcher.core.command_builder import dangerous_run_args, run_args_expose


def _server(**rt):
    base = dict(binary="podman", bind_host="127.0.0.1")
    base.update(rt)
    return Profile(name="p", image="img",
                   runtime=Runtime(**base),
                   mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                   model="/models/m.gguf", settings={"port": 8080})


def _errs(p, **kw):
    return [i.message for i in validate(p, **kw) if i.level == "error"]


def _warns(p, **kw):
    return [i.message for i in validate(p, **kw) if i.level == "warning"]


# -- screening helpers --------------------------------------------------------

def test_dangerous_run_args_flags_escalations():
    assert dangerous_run_args("--privileged")
    assert dangerous_run_args("--entrypoint=/bin/sh")
    assert dangerous_run_args("--cap-add=SYS_ADMIN")
    assert dangerous_run_args("--security-opt label=disable")
    assert dangerous_run_args("--pid=host")
    assert dangerous_run_args("--ipc host")
    assert dangerous_run_args("--userns=host")
    assert dangerous_run_args("-v /:/hostfs:rw")
    assert dangerous_run_args("--volume /etc:/etc")
    assert dangerous_run_args("--device /dev/sda")


def test_dangerous_run_args_allows_benign():
    assert dangerous_run_args("--shm-size=1g") == []
    assert dangerous_run_args("-v /home/me/models:/models:ro") == []
    assert dangerous_run_args("--device nvidia.com/gpu=all") == []   # the launcher's own CDI form
    assert dangerous_run_args("") == []


def test_dangerous_run_args_flags_sensitive_subpaths_and_socket():
    # A sensitive *subpath* and the runtime socket are screened, not only
    # exact top-level dirs; either one is a container escape.
    assert dangerous_run_args("-v /var/run/docker.sock:/var/run/docker.sock")
    assert dangerous_run_args("-v /run/user/1000/podman/podman.sock:/s")
    assert dangerous_run_args("-v /home/me/.ssh:/keys:ro")
    assert dangerous_run_args("--volume /root/.aws:/aws")
    assert dangerous_run_args("--mount type=bind,source=/var/run/docker.sock,target=/s")
    # `..` traversal must not launder a sensitive path past the screen.
    assert dangerous_run_args("-v /tmp/../etc:/x")


def test_dangerous_run_args_still_allows_ordinary_home_data():
    # Legitimate home data mounts must keep working; only credential dotfile
    # dirs under /home are sensitive.
    assert dangerous_run_args("-v /home/me/models:/models:ro") == []
    assert dangerous_run_args("-v /home/me/work/data:/data") == []


def test_dangerous_run_args_flags_runtime_path_flags():
    # An attacker-named OCI runtime / hooks dir is a host-exec vector.
    assert dangerous_run_args("--runtime /tmp/evil")
    assert dangerous_run_args("--hooks-dir /tmp/hooks")


def test_run_args_expose_detects_host_net_and_publish():
    assert run_args_expose("--network host")
    assert run_args_expose("--net=host")
    assert run_args_expose("-p 0.0.0.0:8080:8080")
    assert run_args_expose("--publish 8080:8080")
    assert not run_args_expose("--shm-size=1g")
    assert not run_args_expose("")


# -- validate() integration ---------------------------------------------------

def test_validate_errors_on_privileged_extra_args():
    p = _server(extra_run_args="--privileged -v /:/hostfs:rw")
    errs = _errs(p)
    assert any("extra" in e.lower() and "podman" in e.lower() or "run arg" in e.lower()
               for e in errs)


def test_validate_network_host_requires_key_even_on_loopback_bind():
    # bind_host is loopback, but --network host publishes on every interface.
    p = _server(bind_host="127.0.0.1", extra_run_args="--network host")
    assert any("unauthenticated" in e for e in _errs(p))
    # with a key, no exposure error
    p.settings["api-key"] = "secret"
    assert not any("unauthenticated" in e for e in _errs(p))


def test_validate_blank_api_key_does_not_satisfy_exposure_guard():
    p = _server(bind_host="0.0.0.0")
    p.settings["api-key"] = "   "          # whitespace: dropped from argv, so no real auth
    assert any("unauthenticated" in e for e in _errs(p))


def test_validate_rejects_non_ip_bind_host():
    p = _server(bind_host="evil.example.com")
    assert any("bind" in e.lower() and "address" in e.lower() for e in _errs(p))


def test_validate_accepts_ip_and_loopback_and_wildcard_bind_hosts():
    for h in ("127.0.0.1", "0.0.0.0", "::1", "[::1]", "localhost", "192.168.1.5", "::"):
        p = _server(bind_host=h)
        p.settings["api-key"] = "secret"   # silence the exposure error for non-loopback
        assert not any("bind" in e.lower() and "address" in e.lower() for e in _errs(p)), h


def test_validate_warns_on_sensitive_mount_source():
    p = _server()
    p.mounts.append(Mount(host="/", container="/host", role="data", mode="rw"))
    assert any("sensitive" in w.lower() or "host path" in w.lower() for w in _warns(p))


def test_validate_router_rejects_model_id_with_newline():
    router = Profile(name="r", image="img", runtime=Runtime(binary="podman", bind_host="127.0.0.1"),
                     mode="router",
                     mounts=[Mount(host="/h", container="/models", role="model", mode="ro")],
                     settings={"port": 8080})
    member = RouterMember(profile="m", model_id="bad\n[evil]\nkey = x")
    mp = Profile(name="m", image="img", runtime=Runtime(binary="podman"),
                 model="/models/m.gguf")
    errs = _errs(router, members=((member, mp),))
    assert any("model id" in e.lower() or "invalid" in e.lower() for e in errs)
