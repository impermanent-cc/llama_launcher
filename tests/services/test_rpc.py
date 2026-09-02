import pytest
from pathlib import Path
from llama_launcher.services import rpc
from llama_launcher.core.spec import Profile, Runtime, RpcWorker

@pytest.fixture(autouse=True)
def _reset_tunnels():
    rpc._TUNNELS.clear()
    yield
    rpc._TUNNELS.clear()

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


def test_launch_pool_head_fail_clears_tunnels_entry(tmp_path, monkeypatch):
    # A head that fails to start after every worker is ready
    # must not leave a stale _TUNNELS[profile.name] entry pointing at
    # already-terminated tunnel handles.
    monkeypatch.setattr(rpc, "get_node", lambda base, name:
        {"box2": rpc.Node(name="box2", kind="remote", connection="box2",
                          ssh_target="user@box2")}.get(name))
    p = Profile(name="pool", image="img", model="/m/x.gguf",
                runtime=Runtime(launch_mode="rpc", rpc_workers=[
                    RpcWorker(node="box2", device="CPU", port=50052)]))

    def run(argv):
        return 1 if "--rpc" in argv else 0  # worker starts fine, head fails

    class FakeHandle:
        def terminate(self):
            pass

    res = rpc.launch_pool(p, tmp_path, run=run, popen=lambda argv: FakeHandle(),
                          connect=lambda a: object(), alloc_port=lambda: 41000)
    assert res.ok is False
    assert "pool" not in rpc._TUNNELS


def test_launch_pool_terminates_stale_tunnels_on_relaunch(tmp_path, monkeypatch):
    # Relaunching a pool without an intervening stop_pool must
    # not orphan the previous call's ssh tunnel handles.
    monkeypatch.setattr(rpc, "get_node", lambda base, name:
        {"box2": rpc.Node(name="box2", kind="remote", connection="box2",
                          ssh_target="user@box2")}.get(name))
    p = Profile(name="pool", image="img", model="/m/x.gguf",
                runtime=Runtime(launch_mode="rpc", rpc_workers=[
                    RpcWorker(node="box2", device="CPU", port=50052)]))

    class FakeHandle:
        def __init__(self, gen):
            self.gen = gen
            self.terminated = False

        def terminate(self):
            self.terminated = True

    def make_popen(gen):
        return lambda argv: FakeHandle(gen)

    res1 = rpc.launch_pool(p, tmp_path, run=lambda argv: 0, popen=make_popen("first"),
                           connect=lambda a: object(), alloc_port=lambda: 41000)
    assert res1.ok
    first_handles = list(rpc._TUNNELS["pool"])
    assert first_handles and all(h.gen == "first" for h in first_handles)

    res2 = rpc.launch_pool(p, tmp_path, run=lambda argv: 0, popen=make_popen("second"),
                           connect=lambda a: object(), alloc_port=lambda: 41001)
    assert res2.ok

    assert all(h.terminated for h in first_handles)          # first call's tunnels not leaked
    assert all(h.gen == "second" for h in rpc._TUNNELS["pool"])  # only second call's tunnels live


def test_launch_pool_bad_ssh_target_tears_down_started_worker(tmp_path, monkeypatch):
    # tunnel_argv raising ValueError for a malformed ssh_target
    # (after the worker's container already started) must still tear down the
    # started worker and fail cleanly instead of leaking it.
    monkeypatch.setattr(rpc, "get_node", lambda base, name:
        {"box2": rpc.Node(name="box2", kind="remote", connection="box2",
                          ssh_target="-oProxyCommand=x")}.get(name))
    p = Profile(name="pool", image="img", model="/m/x.gguf",
                runtime=Runtime(launch_mode="rpc", rpc_workers=[
                    RpcWorker(node="box2", device="CPU", port=50052)]))
    order = []

    def run(argv):
        order.append(argv)
        return 0

    res = rpc.launch_pool(p, tmp_path, run=run, popen=lambda argv: None,
                          connect=lambda a: object(), alloc_port=lambda: 41000)
    assert res.ok is False
    # the worker's container-run happened, and _teardown's stop call for it too
    assert any("rpc-server" in " ".join(a) for a in order)
    assert any("stop" in a for a in order)
