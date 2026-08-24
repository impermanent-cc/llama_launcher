from llama_launcher.core.command_builder import build_command, build_rpc_endpoints, build_worker_command
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
    # The RPC head must force the llama-server entrypoint even when the pool
    # image's tag ("b1") doesn't match the full/light heuristic; a pool image is
    # a full-style (tools.sh-entrypoint) build, so the head would otherwise run
    # the dispatcher and print usage instead of serving.
    assert argv[argv.index("--entrypoint") + 1] == "/app/llama-server"


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


def _pool():
    return Profile(name="pool", image="localhost/llama-rpc:b1",
                   runtime=Runtime(launch_mode="rpc", gpu_mode="cdi"))


def test_worker_command_cpu_donor_loopback_publish_no_gpu():
    argv = build_worker_command(_pool(), RpcWorker(node="box2", device="CPU",
                                mem_mb=32000, port=50052), index=1, connection="box2")
    assert argv[:3] == ["podman", "--connection", "box2"]
    assert "-d" in argv and "--device" not in argv and "--gpus" not in argv
    assert "-p" in argv and argv[argv.index("-p") + 1] == "127.0.0.1:50052:50052"
    assert "--name" in argv and argv[argv.index("--name") + 1] == "llama-pool-rpc1"
    assert argv[argv.index("--entrypoint") + 1] == "/app/ggml-rpc-server"
    tail = argv[argv.index("localhost/llama-rpc:b1") + 1:]
    # mem_mb is a preflight-only pledge; current ggml-rpc-server has no --mem flag,
    # so it must NOT reach the worker argv (an unknown arg makes it exit).
    assert "--mem" not in argv
    assert tail == ["-H", "0.0.0.0", "-p", "50052", "-d", "CPU"]


def test_worker_command_gpu_donor_adds_cdi_and_omits_zero_mem():
    argv = build_worker_command(_pool(), RpcWorker(node="local", device="CUDA0",
                                mem_mb=0, port=50053), index=0, connection="")
    assert "--connection" not in argv
    assert argv[argv.index("--device") + 1] == "nvidia.com/gpu=all"
    tail = argv[argv.index("localhost/llama-rpc:b1") + 1:]
    assert tail == ["-H", "0.0.0.0", "-p", "50053", "-d", "CUDA0"]   # no --mem


def test_worker_command_labels_tag_the_pool():
    argv = build_worker_command(_pool(), RpcWorker(node="box2"), index=2, connection="box2")
    joined = " ".join(argv)
    assert "llama-launcher.mode=rpc-worker" in joined
    assert "llama-launcher.pool=pool" in joined
    assert "llama-launcher.profile=pool" in joined
