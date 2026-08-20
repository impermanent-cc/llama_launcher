from llama_launcher.core.spec import Profile, Runtime, RpcWorker
from llama_launcher.services.pool_preflight import gather_donations, headline


def _p():
    return Profile(name="pool", runtime=Runtime(launch_mode="rpc", rpc_workers=[
        RpcWorker(node="local", device="CUDA0", mem_mb=0),
        RpcWorker(node="box2", device="CPU", mem_mb=0)]))


def test_gather_uses_free_when_no_pledge(tmp_path, monkeypatch):
    import llama_launcher.services.pool_preflight as pp
    monkeypatch.setattr(pp, "get_node", lambda base, name:
        type("N", (), {"kind": "local" if name == "local" else "remote",
                       "ssh_target": "user@box2"})())
    gb = 1024 ** 3
    dons = gather_donations(_p(), tmp_path,
        gpus=lambda node: 20 * gb if node == "local" else 0,
        ram=lambda node: 64 * gb if node == "box2" else 0)
    assert ("vram", 20 * gb) in dons and ("ram", 64 * gb) in dons


def test_gather_uses_pledge_when_mem_mb_set(tmp_path, monkeypatch):
    """A worker's --mem pledge wins over the free-memory probe."""
    import llama_launcher.services.pool_preflight as pp
    monkeypatch.setattr(pp, "get_node", lambda base, name: None)
    mib = 1024 * 1024
    profile = Profile(name="pool", runtime=Runtime(launch_mode="rpc", rpc_workers=[
        RpcWorker(node="local", device="CUDA0", mem_mb=8192)]))
    dons = gather_donations(profile, tmp_path,
        gpus=lambda node: 999 * mib, ram=lambda node: 999 * mib)
    assert dons == [("vram", 8192 * mib)]


def test_headline_reports_fit():
    gb = 1024 ** 3
    s = headline(100 * gb, [("vram", 60 * gb), ("ram", 60 * gb)])
    assert "fits" in s.lower() and "120" in s


def test_headline_reports_shortfall():
    gb = 1024 ** 3
    s = headline(200 * gb, [("vram", 60 * gb), ("ram", 60 * gb)])
    assert "does not fit" in s.lower() and "short by" in s.lower()
