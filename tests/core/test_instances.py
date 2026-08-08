from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.core.instances import Instance, build_instances


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


def test_running_first_then_by_name():
    containers = [
        {"name": "llama-b", "running": False, "profile": "b", "mode": "server"},
        {"name": "llama-a", "running": True, "profile": "a", "mode": "server"},
        {"name": "llama-c", "running": True, "profile": "c", "mode": "server"},
    ]
    profiles = [_prof("a", 8080), _prof("b", 8081), _prof("c", 8082)]
    names = [i.name for i in build_instances(containers, profiles)]
    assert names == ["llama-a", "llama-c", "llama-b"]   # running (a,c) before stopped (b)
