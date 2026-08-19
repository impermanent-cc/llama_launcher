import pytest

from llama_launcher.core.instances import Instance
from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _inst(name="llama-emb", profile="emb", port=8081, running=True, embeddings=True):
    return Instance(name=name, profile=profile, mode="server", running=running,
                    port=port, host="127.0.0.1", embeddings=embeddings, reranking=False)


def test_monitored_profile_falls_back_to_current_when_none(win):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    assert win._monitor._active_instance is None
    assert win._monitor._monitored_profile().name == "Solo"
    assert win._monitor._monitored_container_name() == win._container_name()


def test_monitored_profile_resolves_active_instance(win):
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    win._monitor._active_instance = _inst()
    assert win._monitor._monitored_profile().name == "emb"          # not the form's "Solo"
    assert win._monitor._monitored_container_name() == "llama-emb"


def test_instance_summary_embedding_row_is_ready(win, monkeypatch):
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    s = win._monitor.instance_summary(_inst(embeddings=True))
    assert s["health"] == "ready" and s["stat"] == "ready"


def test_instance_summary_gen_row_shows_tok_s(win, monkeypatch):
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    monkeypatch.setattr("llama_launcher.ui.main_window.metrics.fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 64.0})
    s = win._monitor.instance_summary(_inst(embeddings=False))
    assert s["stat"] == "64 tok/s"


def test_build_instances_data_scans_profiles_once_and_builds_rows(monkeypatch):
    """The off-thread builder is a pure function of a primitives-only target: it
    does all the blocking I/O (list_launcher_containers + per-instance probes)
    and scans profiles exactly ONCE for the whole table (folding the old N+1
    per-instance list_profiles scans)."""
    from llama_launcher.ui.controllers import monitor_controller as mc
    profs = [Profile(name="emb", image="img", settings={"port": 8081, "embeddings": True}),
             Profile(name="gen", image="img", settings={"port": 8080})]
    scans = []
    monkeypatch.setattr(mc, "list_profiles", lambda base: (scans.append(base), profs)[1])
    monkeypatch.setattr(mc.runtime, "list_launcher_containers", lambda b: [
        {"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"},
        {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"}])
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(mc.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 42.0})
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    target = {"binary": "podman", "base_dir": "/b", "router_base_dir": "/r"}
    data = mc.build_instances_data(target)
    rows = {r["name"]: r for r in data["rows"]}
    assert rows["llama-emb"]["stat"] == "ready"      # embedding row -> ready, no tok/s
    assert rows["llama-gen"]["stat"] == "42 tok/s"   # gen row -> tok/s from /metrics
    assert len(scans) == 1                           # profiles scanned exactly once
    assert [i.name for i in data["instances"]] == ["llama-emb", "llama-gen"]


def test_build_instances_data_enriches_rows_with_tok_kv_mode(monkeypatch):
    """Each row carries structured tok_s / kv_pct / mode for the stat cards: a gen
    server gets tok/s + KV from /metrics + /slots; an embedding server gets neither."""
    from llama_launcher.ui.controllers import monitor_controller as mc
    profs = [Profile(name="gen", image="img", settings={"port": 8080}),
             Profile(name="emb", image="img", settings={"port": 8081, "embeddings": True})]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)
    monkeypatch.setattr(mc.runtime, "list_launcher_containers", lambda b: [
        {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"},
        {"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"}])
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(mc.metrics, "fetch_metrics",
                        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 64.0})
    monkeypatch.setattr(mc.metrics, "fetch_slots",
                        lambda *a, **k: [{"n_ctx": 100, "n_prompt_tokens_processed": 30}])
    target = {"binary": "podman", "base_dir": "/b", "router_base_dir": "/r"}
    rows = {r["name"]: r for r in mc.build_instances_data(target)["rows"]}
    assert rows["llama-gen"]["tok_s"] == 64.0
    assert abs(rows["llama-gen"]["kv_pct"] - 0.30) < 1e-9   # slots-derived KV
    assert rows["llama-gen"]["mode"] == "server"
    assert rows["llama-emb"]["tok_s"] is None               # embedding: no tok/s
    assert rows["llama-emb"]["kv_pct"] is None


def _two_containers(*_a, **_k):
    return [
        {"name": "llama-solo", "running": True, "profile": "Solo", "mode": "server"},
        {"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"},
    ]


def _populate(win):
    """Drive the off-thread instances gather to completion: tick 1 dispatches,
    the drain lets the pooled task finish, tick 2 renders it into self._instances."""
    from PySide6.QtCore import QThreadPool
    win._monitor._refresh_instances_list()
    QThreadPool.globalInstance().waitForDone(2000)
    win._monitor._refresh_instances_list()


def test_selecting_an_instance_sets_active_without_touching_form(win, monkeypatch):
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        _two_containers)
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    assert win._monitor._active_instance is not None and win._monitor._active_instance.name == "llama-emb"
    assert win._configure_panel.current_profile().name == "Solo"          # form untouched
    assert win._monitor._monitored_profile().name == "emb"        # monitor retargeted


def test_selecting_own_container_clears_active(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        _two_containers)
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    _populate(win)
    win._monitor._on_instance_selected("llama-solo")              # the form's own container
    assert win._monitor._active_instance is None                  # falls back to current profile


def test_instances_rows_rendered_off_thread(win, monkeypatch):
    """The instances table is gathered off the UI thread: the first tick only
    dispatches the pooled gather (no rows yet); a later tick renders the result.

    Mirrors the monitor-summary off-thread cadence so N per-instance probes never
    block the event loop on a Monitor tick.
    """
    from PySide6.QtCore import QThreadPool
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        _two_containers)
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    win._monitor._refresh_instances_list()                 # tick 1: dispatch only
    assert win.monitor_panel.instances_table.rowCount() == 0
    QThreadPool.globalInstance().waitForDone(2000)
    win._monitor._refresh_instances_list()                 # tick 2: renders gathered rows
    assert win.monitor_panel.instances_table.rowCount() == 2


def test_active_instance_cleared_when_its_container_vanishes(win, monkeypatch):
    """A monitored instance whose container disappears on its own (crash, external
    stop) is auto-cleared so the Monitor falls back to the form profile instead of
    stranding on a dead target -- previously only explicit Stop/Remove cleared it."""
    from PySide6.QtCore import QThreadPool
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        _two_containers)
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    QThreadPool.globalInstance().waitForDone(2000)         # drain the gather update_status dispatched
    assert win._monitor._active_instance is not None
    # emb's container vanishes from the fresh list
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        lambda *a, **k: [{"name": "llama-solo", "running": True,
                                          "profile": "Solo", "mode": "server"}])
    _populate(win)                                         # dispatch+render the shrunken list
    assert win._monitor._active_instance is None


def test_active_instance_kept_when_fresh_list_is_empty(win, monkeypatch):
    """A transient `podman ps` failure returns [] (not an error), so an EMPTY
    fresh list is ambiguous -- it must NOT be read as 'my instance vanished' and
    yank the user off the monitored instance + retarget the log follower. Only a
    non-empty list that omits the instance is genuine evidence of external stop."""
    from PySide6.QtCore import QThreadPool
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    monkeypatch.setattr("llama_launcher.ui.main_window.health.probe_health",
                        lambda *a, **k: "ready")
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        _two_containers)
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    QThreadPool.globalInstance().waitForDone(2000)
    assert win._monitor._active_instance is not None
    # ps blip: returns [] even though the instance is really still running
    monkeypatch.setattr("llama_launcher.ui.main_window.runtime.list_launcher_containers",
                        lambda *a, **k: [])
    _populate(win)
    assert win._monitor._active_instance is not None       # not yanked off on an empty list
    assert win._monitor._active_instance.name == "llama-emb"


def test_stop_instance_spawns_stop_argv(win, monkeypatch):
    win._configure_panel.load_profile(Profile(name="Solo", image="img", settings={"port": 8080}))
    spawned = []
    monkeypatch.setattr(win._launch, "_spawn_async",
                        lambda argv, on_done=None, on_error=None: spawned.append(argv))
    win._monitor._on_instance_stop("llama-emb")
    assert spawned and spawned[0][:2] == ["podman", "stop"] and spawned[0][-1] == "llama-emb"
