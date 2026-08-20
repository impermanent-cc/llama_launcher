from llama_launcher.core.command_builder import build_command, build_rpc_endpoints
from llama_launcher.core.spec import Mount, Profile, RpcWorker, Runtime


def test_build_rpc_endpoints_uses_resolver_ports():
    ws = [RpcWorker(node="local", port=50052), RpcWorker(node="box2", port=50052)]
    resolve = {id(ws[0]): 50052, id(ws[1]): 41000}.__getitem__
    got = build_rpc_endpoints(ws, lambda w: resolve(id(w)))
    assert got == "127.0.0.1:50052,127.0.0.1:41000"


def test_build_rpc_endpoints_empty():
    assert build_rpc_endpoints([], lambda w: 0) == ""


def _rpc_profile():
    return Profile(name="pool", image="localhost/llama-rpc:b1",
                   model="/models/big.gguf",
                   mounts=[Mount(host="/models", container="/models")],
                   runtime=Runtime(launch_mode="rpc"))


def test_build_command_rpc_head_is_host_networked_and_local():
    argv = build_command(_rpc_profile(), rpc_endpoints="127.0.0.1:50052",
                         connection="ignored-because-head-is-local")
    assert "--network" in argv and argv[argv.index("--network") + 1] == "host"
    assert "-p" not in argv                     # host networking drops the publish
    assert "--connection" not in argv           # head runs locally
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50052"
    assert argv[0] == "podman" and "run" in argv


def test_build_command_container_mode_unchanged():
    p = Profile(name="plain", image="img", model="/m/x.gguf",
                mounts=[Mount(host="/m", container="/m")])
    argv = build_command(p, connection="box2")
    assert "--network" not in argv
    assert argv[:3] == ["podman", "--connection", "box2"]
    assert "-p" in argv


def test_build_command_rpc_head_binds_to_configured_host_not_all_interfaces():
    # R1: --host must be the profile's configured bind_host (loopback by default
    # convention here), NOT the 0.0.0.0 default from _server_args. Since --network
    # host drops the -p bind_host:port:port translation, --host 0.0.0.0 would
    # expose the head API on every interface on the LAN.
    p = Profile(name="pool", image="localhost/llama-rpc:b1",
                model="/models/big.gguf",
                mounts=[Mount(host="/models", container="/models")],
                runtime=Runtime(launch_mode="rpc", bind_host="127.0.0.1"))
    argv = build_command(p, rpc_endpoints="127.0.0.1:50052")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
