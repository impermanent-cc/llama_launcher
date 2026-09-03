from llama_launcher.core.spec import Profile, Runtime, RpcWorker
from llama_launcher.core.instances import Instance, build_instances, worker_card_title


def _prof(name, port, host="127.0.0.1", embeddings=False, reranking=False):
    settings = {"port": port}
    if embeddings:
        settings["embeddings"] = True
    if reranking:
        settings["reranking"] = True
    return Profile(name=name, runtime=Runtime(bind_host=host), settings=settings)


def test_join_recovers_port_host_endpoints():
    containers = [{"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"}]
    profiles = [_prof("emb", 8081, host="0.0.0.0", embeddings=True)]
    (inst,) = build_instances(containers, profiles)
    assert inst == Instance(name="llama-emb", profile="emb", mode="server",
                            running=True, port=8081, host="127.0.0.1",
                            embeddings=True, reranking=False)


def test_unmatched_container_has_no_port():
    containers = [{"name": "llama-gone", "running": False, "profile": "gone", "mode": "server"}]
    inst, = build_instances(containers, [])
    assert inst.port is None and inst.host == "127.0.0.1" and inst.embeddings is False


def test_carries_stop_timeout_from_profile():
    containers = [{"name": "llama-slow", "running": True, "profile": "slow", "mode": "server"}]
    profiles = [Profile(name="slow", runtime=Runtime(stop_timeout=60), settings={"port": 8080})]
    (inst,) = build_instances(containers, profiles)
    assert inst.stop_timeout == 60


def test_carries_binary_from_profile():
    containers = [{"name": "llama-d", "running": True, "profile": "d", "mode": "server"}]
    profiles = [Profile(name="d", runtime=Runtime(binary="docker"), settings={"port": 8080})]
    (inst,) = build_instances(containers, profiles, "podman")
    assert inst.binary == "docker"   # the profile's binary, not the listing default


def test_unmatched_container_uses_listing_binary():
    containers = [{"name": "llama-gone", "running": False, "profile": "gone", "mode": "server"}]
    (inst,) = build_instances(containers, [], "docker")
    assert inst.binary == "docker"   # falls back to the binary that listed it


def test_unmatched_container_defaults_stop_timeout():
    containers = [{"name": "llama-gone", "running": False, "profile": "gone", "mode": "server"}]
    inst, = build_instances(containers, [])
    assert inst.stop_timeout == 10


def test_running_first_then_by_name():
    containers = [
        {"name": "llama-b", "running": False, "profile": "b", "mode": "server"},
        {"name": "llama-a", "running": True, "profile": "a", "mode": "server"},
        {"name": "llama-c", "running": True, "profile": "c", "mode": "server"},
    ]
    profiles = [_prof("a", 8080), _prof("b", 8081), _prof("c", 8082)]
    names = [i.name for i in build_instances(containers, profiles)]
    assert names == ["llama-a", "llama-c", "llama-b"]   # running (a,c) before stopped (b)


def test_container_rows_default_to_container_kind():
    containers = [{"name": "llama-c", "running": True, "profile": "c", "mode": "server"}]
    (inst,) = build_instances(containers, [_prof("c", 8080)])
    assert inst.kind == "container" and inst.pid is None


def test_native_row_carries_kind_and_pid():
    rows = [{"name": "llama-n", "running": True, "profile": "n", "mode": "server",
             "kind": "native", "pid": 4242}]
    (inst,) = build_instances(rows, [_prof("n", 8080)])
    assert inst.kind == "native" and inst.pid == 4242


def test_rpc_worker_device_resolved_by_name_index():
    """A worker container shares its pool head's `llama-launcher.profile` label
    label, so its device comes from the profile's rpc_workers list, indexed
    by the `-rpcN` suffix on the container's own name."""
    containers = [
        {"name": "llama-pool-rpc0", "running": True, "profile": "pool", "mode": "rpc-worker"},
        {"name": "llama-pool-rpc1", "running": True, "profile": "pool", "mode": "rpc-worker"},
    ]
    profiles = [Profile(name="pool", settings={"port": 8080}, runtime=Runtime(
        launch_mode="rpc",
        rpc_workers=[RpcWorker(node="box1", device="CPU"), RpcWorker(node="box2", device="CUDA0")]))]
    insts = {i.name: i for i in build_instances(containers, profiles)}
    assert insts["llama-pool-rpc0"].device == "CPU"
    assert insts["llama-pool-rpc1"].device == "CUDA0"


def test_rpc_worker_device_empty_when_index_out_of_range():
    containers = [{"name": "llama-pool-rpc5", "running": True, "profile": "pool", "mode": "rpc-worker"}]
    profiles = [Profile(name="pool", settings={"port": 8080}, runtime=Runtime(
        launch_mode="rpc", rpc_workers=[RpcWorker(node="box1")]))]
    (inst,) = build_instances(containers, profiles)
    assert inst.device == ""


def test_rpc_worker_device_empty_when_no_name_suffix():
    containers = [{"name": "llama-pool-worker", "running": True, "profile": "pool", "mode": "rpc-worker"}]
    profiles = [Profile(name="pool", settings={"port": 8080}, runtime=Runtime(
        launch_mode="rpc", rpc_workers=[RpcWorker(node="box1", device="CUDA0")]))]
    (inst,) = build_instances(containers, profiles)
    assert inst.device == ""


def test_non_worker_instance_has_no_device():
    containers = [{"name": "llama-a", "running": True, "profile": "a", "mode": "server"}]
    (inst,) = build_instances(containers, [_prof("a", 8080)])
    assert inst.device == ""


def test_worker_card_title_includes_node():
    inst = Instance(name="llama-pool-rpc1", profile="pool", mode="rpc-worker", running=True,
                    port=None, host="127.0.0.1", embeddings=False, reranking=False, node="box2")
    assert worker_card_title(inst) == "rpc-worker \u00b7 box2"


def test_worker_card_title_includes_device_when_resolved():
    inst = Instance(name="llama-pool-rpc1", profile="pool", mode="rpc-worker", running=True,
                    port=None, host="127.0.0.1", embeddings=False, reranking=False,
                    node="box2", device="CUDA0")
    assert worker_card_title(inst) == "rpc-worker \u00b7 box2 \u00b7 CUDA0"
