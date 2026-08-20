from llama_launcher.core.command_builder import build_rpc_endpoints
from llama_launcher.core.spec import RpcWorker


def test_build_rpc_endpoints_uses_resolver_ports():
    ws = [RpcWorker(node="local", port=50052), RpcWorker(node="box2", port=50052)]
    resolve = {id(ws[0]): 50052, id(ws[1]): 41000}.__getitem__
    got = build_rpc_endpoints(ws, lambda w: resolve(id(w)))
    assert got == "127.0.0.1:50052,127.0.0.1:41000"


def test_build_rpc_endpoints_empty():
    assert build_rpc_endpoints([], lambda w: 0) == ""
