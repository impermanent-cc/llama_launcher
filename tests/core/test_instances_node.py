from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.core.instances import build_instances


def _rows():
    return [{"name": "llama-p", "running": True, "profile": "p", "mode": "server"}]


def _profiles():
    return [Profile(name="p", runtime=Runtime(bind_host="0.0.0.0"),
                    settings={"port": 9000})]


def test_local_instances_tagged_local_and_dial_loopback():
    inst = build_instances(_rows(), _profiles())[0]
    assert inst.node == "local"
    assert inst.host == "127.0.0.1"          # dial_host maps 0.0.0.0 -> loopback


def test_remote_instances_tagged_and_use_node_host():
    inst = build_instances(_rows(), _profiles(), node="box-b",
                           node_host="192.168.1.11")[0]
    assert inst.node == "box-b"
    assert inst.host == "192.168.1.11"        # dial the worker's LAN IP, not loopback
