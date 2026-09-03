from llama_launcher.core.spec import RpcWorker
from llama_launcher.ui.widgets.rpc_workers_table import RpcWorkersTable


def test_table_round_trips_workers(qtbot):
    t = RpcWorkersTable(node_names=["local", "box2"])
    qtbot.addWidget(t)
    ws = [
        RpcWorker(node="local", device="CUDA0", mem_mb=8000, port=50052),
        RpcWorker(node="box2", device="CPU", mem_mb=32000, port=50052),
    ]
    t.set_workers(ws)
    assert t.workers() == ws


def test_starts_empty(qtbot):
    t = RpcWorkersTable(node_names=["local"])
    qtbot.addWidget(t)
    assert t.workers() == []


def test_add_blank_row_and_remove(qtbot):
    t = RpcWorkersTable(node_names=["local", "box2"])
    qtbot.addWidget(t)
    t._add_blank()
    assert len(t.workers()) == 1
    t.table.selectRow(0)
    t._remove_selected()
    assert t.workers() == []


def test_changed_signal_emitted_on_set_workers(qtbot):
    t = RpcWorkersTable(node_names=["local"])
    qtbot.addWidget(t)
    with qtbot.waitSignal(t.changed, timeout=1000):
        t.set_workers([RpcWorker(node="local")])
