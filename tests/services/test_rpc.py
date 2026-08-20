import pytest
from pathlib import Path
from llama_launcher.services import rpc
from llama_launcher.core.spec import Profile, Runtime, RpcWorker

def test_tunnel_argv_is_guarded_and_loopback():
    argv = rpc.tunnel_argv("user@box2", 41000, 50052)
    assert "-L" in argv and "127.0.0.1:41000:127.0.0.1:50052" in argv
    assert argv[-1] == "user@box2"
    with pytest.raises(ValueError):
        rpc.tunnel_argv("-oProxyCommand=x", 1, 2)

def test_wait_ready_true_when_connect_succeeds():
    calls = {"n": 0}
    def connect(addr):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("refused")
        return object()
    assert rpc.wait_ready(50052, connect, attempts=5, delay=0) is True

def test_wait_ready_false_when_never_connects():
    def connect(addr):
        raise OSError("refused")
    assert rpc.wait_ready(50052, connect, attempts=3, delay=0) is False

def test_launch_pool_starts_workers_before_head(tmp_path, monkeypatch):
    # a local + a remote worker
    from llama_launcher.store import nodes as nodes_store
    monkeypatch.setattr(rpc, "get_node", lambda base, name:
        {"local": rpc.LOCAL_NODE,
         "box2": rpc.Node(name="box2", kind="remote", connection="box2",
                          ssh_target="user@box2")}.get(name))
    p = Profile(name="pool", image="img", model="/m/x.gguf",
                runtime=Runtime(launch_mode="rpc", rpc_workers=[
                    RpcWorker(node="local", device="CUDA0", port=50052),
                    RpcWorker(node="box2", device="CPU", port=50052)]))
    order = []
    def run(argv): order.append(argv); return 0
    popen = lambda argv: order.append(("tunnel", argv))
    res = rpc.launch_pool(p, tmp_path, run=run, popen=popen,
                          connect=lambda a: object(),
                          alloc_port=lambda: 41000)
    assert res.ok
    head_idx = next(i for i, a in enumerate(order)
                    if isinstance(a, list) and "--rpc" in a)
    worker_idxs = [i for i, a in enumerate(order)
                   if isinstance(a, list) and "rpc-server" in " ".join(a)]
    assert worker_idxs and max(worker_idxs) < head_idx      # workers first
    assert "127.0.0.1:50052,127.0.0.1:41000" in " ".join(order[head_idx])

def test_stop_pool_stops_head_then_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(rpc, "get_node", lambda base, name:
        {"local": rpc.LOCAL_NODE,
         "box2": rpc.Node(name="box2", kind="remote", connection="box2",
                          ssh_target="user@box2")}.get(name))
    p = Profile(name="pool", image="img",
                runtime=Runtime(launch_mode="rpc", rpc_workers=[
                    RpcWorker(node="box2", device="CPU")]))
    order = []
    rpc.stop_pool(p, tmp_path, run=lambda argv: order.append(argv))
    names = [" ".join(a) for a in order]
    assert any("llama-pool" in n and "rpc" not in n for n in names[:1])  # head first
    assert any("llama-pool-rpc" in n for n in names)
