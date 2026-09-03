import pytest

from llama_launcher.core.instances import Instance
from llama_launcher.core.spec import Profile, Runtime
from llama_launcher.services import rpc
from llama_launcher.store import profiles as store
from llama_launcher.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _inst(name="llama-emb", profile="emb", port=8081, running=True, embeddings=True):
    return Instance(
        name=name,
        profile=profile,
        mode="server",
        running=running,
        port=port,
        host="127.0.0.1",
        embeddings=embeddings,
        reranking=False,
    )


def test_monitored_profile_falls_back_to_current_when_none(win):
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    assert win._monitor._active_instance is None
    assert win._monitor._monitored_profile().name == "Solo"
    assert win._monitor._monitored_container_name() == win._container_name()


def test_monitored_profile_resolves_active_instance(win):
    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._active_instance = _inst()
    assert win._monitor._monitored_profile().name == "emb"  # not the form's "Solo"
    assert win._monitor._monitored_container_name() == "llama-emb"


def test_instance_summary_embedding_row_is_ready(win, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    s = win._monitor.instance_summary(_inst(embeddings=True))
    assert s["health"] == "ready" and s["stat"] == "ready"


def test_instance_summary_gen_row_shows_tok_s(win, monkeypatch):
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.metrics.fetch_metrics",
        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 64.0},
    )
    s = win._monitor.instance_summary(_inst(embeddings=False))
    assert s["stat"] == "64 tok/s"


def test_build_instances_data_scans_profiles_once_and_builds_rows(monkeypatch):
    """The off-thread builder is a pure function of a primitives-only target: it
    does all the blocking I/O (list_launcher_containers + per-instance probes)
    and scans profiles exactly ONCE for the whole table."""
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [
        Profile(name="emb", image="img", settings={"port": 8081, "embeddings": True}),
        Profile(name="gen", image="img", settings={"port": 8080}),
    ]
    scans = []
    monkeypatch.setattr(
        mc, "list_profiles", lambda base: (scans.append(base), profs)[1]
    )
    monkeypatch.setattr(
        mc.runtime,
        "list_launcher_containers",
        lambda b: [
            {"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"},
            {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"},
        ],
    )
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(
        mc.metrics,
        "fetch_metrics",
        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 42.0},
    )
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    target = {"binary": "podman", "base_dir": "/b", "router_base_dir": "/r"}
    data = mc.build_instances_data(target)
    rows = {r["name"]: r for r in data["rows"]}
    assert rows["llama-emb"]["stat"] == "ready"  # embedding row -> ready, no tok/s
    assert rows["llama-gen"]["stat"] == "42 tok/s"  # gen row -> tok/s from /metrics
    assert len(scans) == 1  # profiles scanned exactly once
    assert [i.name for i in data["instances"]] == ["llama-emb", "llama-gen"]


def test_build_instances_data_enriches_rows_with_tok_kv_mode(monkeypatch):
    """Each row carries structured tok_s / kv_pct / mode for the stat cards: a gen
    server gets tok/s + KV from /metrics + /slots; an embedding server gets neither."""
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [
        Profile(name="gen", image="img", settings={"port": 8080}),
        Profile(name="emb", image="img", settings={"port": 8081, "embeddings": True}),
    ]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)
    monkeypatch.setattr(
        mc.runtime,
        "list_launcher_containers",
        lambda b: [
            {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"},
            {"name": "llama-emb", "running": True, "profile": "emb", "mode": "server"},
        ],
    )
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(
        mc.metrics,
        "fetch_metrics",
        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 64.0},
    )
    monkeypatch.setattr(
        mc.metrics,
        "fetch_slots",
        lambda *a, **k: [{"n_ctx": 100, "n_prompt_tokens_processed": 30}],
    )
    target = {"binary": "podman", "base_dir": "/b", "router_base_dir": "/r"}
    rows = {r["name"]: r for r in mc.build_instances_data(target)["rows"]}
    assert rows["llama-gen"]["tok_s"] == 64.0
    assert abs(rows["llama-gen"]["kv_pct"] - 0.30) < 1e-9  # slots-derived KV
    assert rows["llama-gen"]["mode"] == "server"
    assert rows["llama-emb"]["tok_s"] is None  # embedding: no tok/s
    assert rows["llama-emb"]["kv_pct"] is None


def test_build_instances_data_titles_rpc_worker_row(monkeypatch):
    """A worker container shares its pool head's `llama-launcher.profile` label
    so it joins to the SAME stored profile as the head; without a title
    override its row would show the head's own name/port. The worker row
    instead gets its own "rpc-worker \u00b7 <node> \u00b7 <device>" title and no port."""
    from llama_launcher.core.spec import RpcWorker
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [
        Profile(
            name="pool",
            image="img",
            runtime=Runtime(
                launch_mode="rpc", rpc_workers=[RpcWorker(node="box2", device="CUDA0")]
            ),
            settings={"port": 8080},
        )
    ]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)

    def _containers(binary, connection=""):
        # The worker's own row is only listed under ITS node's connection
        # (list_launcher_containers is called once per registered node), so
        # inst.node ends up "box2" -- exactly like a real remote worker.
        if connection == "box2":
            return [
                {
                    "name": "llama-pool-rpc0",
                    "running": True,
                    "profile": "pool",
                    "mode": "rpc-worker",
                }
            ]
        return [
            {"name": "llama-pool", "running": True, "profile": "pool", "mode": "server"}
        ]

    monkeypatch.setattr(mc.runtime, "list_launcher_containers", _containers)
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(mc.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    target = {
        "binary": "podman",
        "base_dir": "/b",
        "router_base_dir": "/r",
        "nodes": [
            {
                "name": "local",
                "connection": "",
                "host": "",
                "binary": "podman",
                "enabled": True,
            },
            {
                "name": "box2",
                "connection": "box2",
                "host": "10.0.0.2",
                "binary": "podman",
                "enabled": True,
            },
        ],
    }
    rows = {r["name"]: r for r in mc.build_instances_data(target)["rows"]}
    worker = rows["llama-pool-rpc0"]
    assert "rpc-worker" in worker["profile"] and "box2" in worker["profile"]
    assert "CUDA0" in worker["profile"]
    assert worker["port"] is None
    head = rows["llama-pool"]
    assert head["profile"] == "pool" and head["port"] == 8080  # head's row unaffected


def test_rpc_worker_row_does_not_show_head_http_metrics(monkeypatch):
    """A worker shares the head's port (8080), so an HTTP probe of the worker's
    row hits the HEAD's endpoint. The worker card must NOT surface those metrics
    (an rpc-server has no HTTP endpoint) -- only up/down."""
    from llama_launcher.core.spec import RpcWorker
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [
        Profile(
            name="pool",
            image="img",
            runtime=Runtime(
                launch_mode="rpc", rpc_workers=[RpcWorker(node="box2", device="CUDA0")]
            ),
            settings={"port": 8080},
        )
    ]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)

    def _containers(binary, connection=""):
        if connection == "box2":
            return [
                {
                    "name": "llama-pool-rpc0",
                    "running": True,
                    "profile": "pool",
                    "mode": "rpc-worker",
                }
            ]
        return [
            {"name": "llama-pool", "running": True, "profile": "pool", "mode": "server"}
        ]

    monkeypatch.setattr(mc.runtime, "list_launcher_containers", _containers)
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    # The head IS serving 42 tok/s; the worker row must ignore it.
    monkeypatch.setattr(
        mc.metrics,
        "fetch_metrics",
        lambda *a, **k: {"llamacpp:predicted_tokens_seconds": 42.0},
    )
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    target = {
        "binary": "podman",
        "base_dir": "/b",
        "router_base_dir": "/r",
        "nodes": [
            {
                "name": "local",
                "connection": "",
                "host": "",
                "binary": "podman",
                "enabled": True,
            },
            {
                "name": "box2",
                "connection": "box2",
                "host": "10.0.0.2",
                "binary": "podman",
                "enabled": True,
            },
        ],
    }
    rows = {r["name"]: r for r in mc.build_instances_data(target)["rows"]}
    worker = rows["llama-pool-rpc0"]
    assert worker["tok_s"] is None and worker["stat"] == ""  # no head metrics
    assert worker["health"] == "ready"  # up/down only
    assert rows["llama-pool"]["tok_s"] == 42.0  # head still shows its own


def test_build_instances_data_includes_native_rows(tmp_path, monkeypatch):
    """A native (non-container) server registered under base_dir shows up as a
    Monitor card alongside container rows, carrying its own kind/pid."""
    from llama_launcher.ui.controllers import monitor_controller as mc

    monkeypatch.setattr(mc.runtime, "list_launcher_containers", lambda b: [])
    monkeypatch.setattr(
        mc.native,
        "list_native_instances",
        lambda base: [
            {
                "name": "llama-nat",
                "running": True,
                "profile": "nat",
                "mode": "server",
                "kind": "native",
                "pid": 4242,
            }
        ],
    )
    monkeypatch.setattr(
        mc, "list_profiles", lambda base: [Profile(name="nat", settings={"port": 8080})]
    )
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(mc.metrics, "fetch_metrics", lambda *a, **k: {})
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    target = {
        "binary": "podman",
        "base_dir": str(tmp_path),
        "router_base_dir": str(tmp_path),
    }
    out = mc.build_instances_data(target)
    assert [i.name for i in out["instances"]] == ["llama-nat"]
    assert out["instances"][0].kind == "native" and out["instances"][0].pid == 4242


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
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        _two_containers,
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    assert (
        win._monitor._active_instance is not None
        and win._monitor._active_instance.name == "llama-emb"
    )
    assert win._configure_panel.current_profile().name == "Solo"  # form untouched
    assert win._monitor._monitored_profile().name == "emb"  # monitor retargeted


def test_selecting_own_container_clears_active(win, monkeypatch):
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        _two_containers,
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    _populate(win)
    win._monitor._on_instance_selected("llama-solo")  # the form's own container
    assert win._monitor._active_instance is None  # falls back to current profile


def test_instances_rows_rendered_off_thread(win, monkeypatch):
    """The instances table is gathered off the UI thread: the first tick only
    dispatches the pooled gather (no rows yet); a later tick renders the result.

    Same cadence as the monitor summary, so N per-instance probes never block
    the event loop on a Monitor tick.
    """
    from PySide6.QtCore import QThreadPool

    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        _two_containers,
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    win._monitor._refresh_instances_list()  # tick 1: dispatch only
    assert len(win.monitor_panel.card_names()) == 0
    QThreadPool.globalInstance().waitForDone(2000)
    win._monitor._refresh_instances_list()  # tick 2: renders gathered rows
    assert win.monitor_panel.card_names() == ["llama-emb", "llama-solo"]


def test_active_instance_cleared_when_its_container_vanishes(win, monkeypatch):
    """A monitored instance whose container disappears on its own (crash, external
    stop) is auto-cleared so the Monitor falls back to the form profile instead of
    stranding on a dead target."""
    from PySide6.QtCore import QThreadPool

    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        _two_containers,
    )
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    QThreadPool.globalInstance().waitForDone(
        2000
    )  # drain the gather update_status dispatched
    assert win._monitor._active_instance is not None
    # emb's container vanishes from the fresh list
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        lambda *a, **k: [
            {"name": "llama-solo", "running": True, "profile": "Solo", "mode": "server"}
        ],
    )
    _populate(win)  # dispatch+render the shrunken list
    assert win._monitor._active_instance is None


def test_active_instance_kept_when_fresh_list_is_empty(win, monkeypatch):
    """A transient `podman ps` failure returns [] (not an error), so an EMPTY
    fresh list is ambiguous -- it must NOT be read as 'my instance vanished' and
    yank the user off the monitored instance + retarget the log follower. Only a
    non-empty list that omits the instance is genuine evidence of external stop."""
    from PySide6.QtCore import QThreadPool

    base = store.default_base_dir()
    store.save_profile(Profile(name="emb", image="img", settings={"port": 8081}), base)
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.health.probe_health", lambda *a, **k: "ready"
    )
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        _two_containers,
    )
    _populate(win)
    win._monitor._on_instance_selected("llama-emb")
    QThreadPool.globalInstance().waitForDone(2000)
    assert win._monitor._active_instance is not None
    # ps blip: returns [] even though the instance is really still running
    monkeypatch.setattr(
        "llama_launcher.ui.main_window.runtime.list_launcher_containers",
        lambda *a, **k: [],
    )
    _populate(win)
    assert win._monitor._active_instance is not None  # not yanked off on an empty list
    assert win._monitor._active_instance.name == "llama-emb"


def test_stop_instance_spawns_stop_argv(win, monkeypatch):
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-emb")
    assert (
        spawned
        and spawned[0][:2] == ["podman", "stop"]
        and spawned[0][-1] == "llama-emb"
    )


def test_stop_instance_uses_instance_stop_timeout(win, monkeypatch):
    """A card \u25a0 Stop resolves the container's own grace period from the built
    instance, so each instance honors its profile's stop_timeout."""
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = [
        Instance(
            name="llama-slow",
            profile="slow",
            mode="server",
            running=True,
            port=8081,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            stop_timeout=40,
        )
    ]
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-slow")
    assert spawned[0] == ["podman", "stop", "-t", "40", "llama-slow"]


def test_stop_instance_uses_instance_binary(win, monkeypatch):
    """The card \u25a0 Stop controls the container with ITS profile's binary, not
    whatever binary the currently-loaded form profile happens to use."""
    win._configure_panel.load_profile(
        Profile(
            name="Solo",
            image="img",
            runtime=Runtime(binary="podman"),
            settings={"port": 8080},
        )
    )
    win._monitor._instances = [
        Instance(
            name="llama-dock",
            profile="dock",
            mode="server",
            running=True,
            port=8081,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="docker",
        )
    ]
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-dock")
    assert spawned[0][0] == "docker"


def test_remove_instance_uses_instance_binary(win, monkeypatch):
    """The card \u25a0 Remove uses the stopped container's own binary, not the form's."""
    win._configure_panel.load_profile(
        Profile(
            name="Solo",
            image="img",
            runtime=Runtime(binary="podman"),
            settings={"port": 8080},
        )
    )
    win._monitor._instances = [
        Instance(
            name="llama-dock",
            profile="dock",
            mode="server",
            running=False,
            port=None,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="docker",
        )
    ]
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_remove("llama-dock")
    assert spawned[0][0] == "docker" and spawned[0][-1] == "llama-dock"


def test_stop_instance_threads_remote_node_connection(win, monkeypatch):
    """A card [Stop] on a REMOTE instance must target that node's podman
    --connection, never local podman -- otherwise Stop is silently a no-op
    against the wrong host (or worse, hits a same-named LOCAL container)."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        win.base_dir(),
    )
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = [
        Instance(
            name="llama-rem",
            profile="rem",
            mode="server",
            running=True,
            port=8081,
            host="10.0.0.2",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="podman",
            node="box-b",
        )
    ]
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-rem")
    assert spawned[0] == [
        "podman",
        "--connection",
        "box-b",
        "stop",
        "-t",
        "10",
        "llama-rem",
    ]


def test_remove_instance_threads_remote_node_connection(win, monkeypatch):
    """A card [Remove] on a REMOTE instance must target that node's podman
    --connection, never local podman."""
    from llama_launcher.core.nodes import Node
    from llama_launcher.store.nodes import add_node

    add_node(
        Node(name="box-b", kind="remote", connection="box-b", ssh_target="me@10.0.0.2"),
        win.base_dir(),
    )
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = [
        Instance(
            name="llama-rem",
            profile="rem",
            mode="server",
            running=False,
            port=None,
            host="10.0.0.2",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="podman",
            node="box-b",
        )
    ]
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_remove("llama-rem")
    assert spawned[0] == ["podman", "--connection", "box-b", "rm", "-f", "llama-rem"]


def test_stop_unknown_instance_falls_back_to_default_timeout(win, monkeypatch):
    """If the name isn't in the built instances (transient list gap), stop still
    works with the 10s default rather than crashing."""
    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = []
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-ghost")
    assert spawned[0] == ["podman", "stop", "-t", "10", "llama-ghost"]


def test_native_instance_stop_sends_sigterm_then_schedules_kill(win, monkeypatch):
    import signal

    from llama_launcher.ui.controllers import monitor_controller as mc

    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = [
        Instance(
            name="llama-nat",
            profile="nat",
            mode="server",
            running=True,
            port=8080,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            stop_timeout=7,
            binary="podman",
            kind="native",
            pid=4242,
        )
    ]
    signals = []
    monkeypatch.setattr(
        mc.native, "stop_native", lambda pid, sig: signals.append((pid, sig))
    )
    scheduled = {}
    monkeypatch.setattr(
        mc,
        "_schedule_sigkill",
        lambda pid, delay: scheduled.setdefault("v", (pid, delay)),
        raising=False,
    )
    win._monitor._on_instance_stop("llama-nat")
    assert signals == [(4242, signal.SIGTERM)]
    assert scheduled["v"] == (4242, 7)


def test_stop_head_of_rpc_pool_calls_coordinated_stop_pool(win, monkeypatch):
    """A card [Stop] on the HEAD of an rpc pool (runtime.launch_mode == "rpc")
    performs the coordinated pool stop -- head + every worker -- instead of a
    bare `podman stop` on just the head container. The stop is dispatched OFF
    the UI thread via LaunchController's pool seam; the sync seam here runs it
    inline so the assertion is deterministic."""
    from llama_launcher.core.spec import RpcWorker

    store.save_profile(
        Profile(
            name="pool",
            image="img",
            runtime=Runtime(launch_mode="rpc", rpc_workers=[RpcWorker(node="box2")]),
            settings={"port": 8080},
        ),
        win.base_dir(),
    )
    win._monitor._instances = [
        Instance(
            name="llama-pool",
            profile="pool",
            mode="server",
            running=True,
            port=8080,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
        )
    ]
    called = {}
    monkeypatch.setattr(
        rpc, "stop_pool", lambda p, base, **k: called.setdefault("profile", p)
    )
    monkeypatch.setattr(
        win._launch, "_run_pool_async", lambda work, on_done: on_done(work())
    )
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-pool")
    assert called["profile"].name == "pool"
    assert spawned == []  # NOT a bare container stop


def test_stop_rpc_worker_instance_stops_only_that_worker(win, monkeypatch):
    """A card [Stop] on a WORKER instance (mode "rpc-worker") must stop just that
    one worker container -- never the coordinated pool stop -- even though the
    worker shares its pool head's profile (and that profile is launch_mode "rpc")."""
    from llama_launcher.core.spec import RpcWorker

    store.save_profile(
        Profile(
            name="pool",
            image="img",
            runtime=Runtime(launch_mode="rpc", rpc_workers=[RpcWorker(node="box2")]),
            settings={"port": 8080},
        ),
        win.base_dir(),
    )
    win._monitor._instances = [
        Instance(
            name="llama-pool-rpc0",
            profile="pool",
            mode="rpc-worker",
            running=True,
            port=None,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            node="box2",
        )
    ]
    pool_called = []
    monkeypatch.setattr(rpc, "stop_pool", lambda p, base, **k: pool_called.append(p))
    spawned = []
    monkeypatch.setattr(
        win._launch,
        "_spawn_async",
        lambda argv, on_done=None, on_error=None: spawned.append(argv),
    )
    win._monitor._on_instance_stop("llama-pool-rpc0")
    assert pool_called == []
    assert spawned and spawned[0][-1] == "llama-pool-rpc0"


def test_sigkill_if_alive_kills_when_still_our_process(monkeypatch):
    import signal

    from llama_launcher.ui.controllers import monitor_controller as mc

    monkeypatch.setattr(
        mc.native,
        "read_entries",
        lambda base: [{"pid": 4242, "binary": "/opt/llama-server"}],
    )
    monkeypatch.setattr(mc.native, "is_alive", lambda pid, binary: True)
    killed = []
    monkeypatch.setattr(
        mc.native, "stop_native", lambda pid, sig: killed.append((pid, sig))
    )
    mc._sigkill_if_alive(4242)
    assert killed == [(4242, signal.SIGKILL)]


def test_sigkill_if_alive_skips_when_registry_entry_gone(monkeypatch):
    # A running native process is never pruned, so a missing entry => already dead.
    from llama_launcher.ui.controllers import monitor_controller as mc

    monkeypatch.setattr(mc.native, "read_entries", lambda base: [])
    monkeypatch.setattr(mc.native, "is_alive", lambda pid, binary: True)  # unreached
    killed = []
    monkeypatch.setattr(
        mc.native, "stop_native", lambda pid, sig: killed.append((pid, sig))
    )
    mc._sigkill_if_alive(4242)
    assert killed == []


def test_sigkill_if_alive_skips_when_pid_recycled(monkeypatch):
    # Entry exists but the pid no longer references our binary (OS recycled it).
    from llama_launcher.ui.controllers import monitor_controller as mc

    monkeypatch.setattr(
        mc.native,
        "read_entries",
        lambda base: [{"pid": 4242, "binary": "/opt/llama-server"}],
    )
    monkeypatch.setattr(mc.native, "is_alive", lambda pid, binary: False)
    killed = []
    monkeypatch.setattr(
        mc.native, "stop_native", lambda pid, sig: killed.append((pid, sig))
    )
    mc._sigkill_if_alive(4242)
    assert killed == []


def test_native_instance_remove_calls_native(win, monkeypatch):
    from llama_launcher.ui.controllers import monitor_controller as mc

    win._configure_panel.load_profile(
        Profile(name="Solo", image="img", settings={"port": 8080})
    )
    win._monitor._instances = [
        Instance(
            name="llama-nat",
            profile="nat",
            mode="server",
            running=False,
            port=None,
            host="127.0.0.1",
            embeddings=False,
            reranking=False,
            stop_timeout=10,
            binary="podman",
            kind="native",
            pid=None,
        )
    ]
    removed = []
    monkeypatch.setattr(
        mc.native, "remove_native", lambda name, base: removed.append(name)
    )
    win._monitor._on_instance_remove("llama-nat")
    assert removed == ["llama-nat"]


def test_build_instances_data_live_headline_from_decode_delta(monkeypatch):
    """Card headlines must go live: with a decode baseline for the row, the
    n_decode_total delta rate wins over the completion gauge (which reads 0
    for the whole of an in-flight generation)."""
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [Profile(name="gen", image="img", settings={"port": 8080})]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)
    monkeypatch.setattr(
        mc.runtime,
        "list_launcher_containers",
        lambda b: [
            {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"}
        ],
    )
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(
        mc.metrics,
        "fetch_metrics",
        lambda *a, **k: {
            "llamacpp:n_decode_total": 123.0,
            "llamacpp:predicted_tokens_seconds": 0.0,
        },
    )
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    monkeypatch.setattr(mc.time, "monotonic", lambda: 11.0)
    target = {
        "binary": "podman",
        "base_dir": "/b",
        "router_base_dir": "/r",
        "decode_prev_by_key": {"local/llama-gen": (100.0, 10.0)},
    }
    data = mc.build_instances_data(target)
    rows = {r["name"]: r for r in data["rows"]}
    assert rows["llama-gen"]["stat"] == "23 tok/s"
    assert rows["llama-gen"]["tok_s"] == 23.0
    assert data["decode_now_by_key"]["local/llama-gen"] == (123.0, 11.0)


def test_build_instances_data_headline_falls_back_to_gauge_when_idle(monkeypatch):
    """No counter movement (idle) -> show the last completed request's gauge,
    so an idle card still tells the user how fast the last run was."""
    from llama_launcher.ui.controllers import monitor_controller as mc

    profs = [Profile(name="gen", image="img", settings={"port": 8080})]
    monkeypatch.setattr(mc, "list_profiles", lambda base: profs)
    monkeypatch.setattr(
        mc.runtime,
        "list_launcher_containers",
        lambda b: [
            {"name": "llama-gen", "running": True, "profile": "gen", "mode": "server"}
        ],
    )
    monkeypatch.setattr(mc.health, "probe_health", lambda *a, **k: "ready")
    monkeypatch.setattr(
        mc.metrics,
        "fetch_metrics",
        lambda *a, **k: {
            "llamacpp:n_decode_total": 100.0,
            "llamacpp:predicted_tokens_seconds": 64.0,
        },
    )
    monkeypatch.setattr(mc.metrics, "fetch_slots", lambda *a, **k: [])
    monkeypatch.setattr(mc.time, "monotonic", lambda: 11.0)
    target = {
        "binary": "podman",
        "base_dir": "/b",
        "router_base_dir": "/r",
        "decode_prev_by_key": {"local/llama-gen": (100.0, 10.0)},
    }
    rows = {r["name"]: r for r in mc.build_instances_data(target)["rows"]}
    assert rows["llama-gen"]["stat"] == "64 tok/s"


def test_instances_target_carries_decode_baselines(win):
    win._monitor._cards_decode_prev = {"local/llama-gen": (100.0, 10.0)}
    t = win._monitor._instances_target()
    assert t["decode_prev_by_key"] == {"local/llama-gen": (100.0, 10.0)}


def test_render_instances_advances_decode_baselines(win):
    win._monitor._instances_result = {
        "instances": [],
        "rows": [],
        "decode_now_by_key": {"local/x": (5.0, 1.0)},
    }
    win._monitor._render_instances()
    assert win._monitor._cards_decode_prev == {"local/x": (5.0, 1.0)}


def test_pool_instance_stop_dispatches_off_thread(win, tmp_path):
    # The Monitor instance-card Stop for an RPC pool head must go through
    # LaunchController's off-thread pool seam, not call rpc.stop_pool inline on
    # the UI thread (per-worker `podman stop` + ssh teardown would freeze the GUI).
    base = win.base_dir()
    store.save_profile(
        Profile(name="pool", image="img:tag", runtime=Runtime(launch_mode="rpc")), base
    )
    head = Instance(
        name="llama-pool",
        profile="pool",
        mode="server",
        running=True,
        port=8080,
        host="127.0.0.1",
        embeddings=False,
        reranking=False,
    )
    win._monitor._instances = [head]

    import llama_launcher.services.rpc as rpc

    sync_called = {}

    def monkeypatch_stop(p, base, **k):
        sync_called.setdefault("ok", True)

    rpc_stop_orig = rpc.stop_pool
    rpc.stop_pool = monkeypatch_stop
    delegated = {}
    win._launch.stop_pool_async = lambda profile: delegated.setdefault(
        "name", profile.name
    )
    try:
        win._monitor._on_instance_stop("llama-pool")
    finally:
        rpc.stop_pool = rpc_stop_orig

    assert delegated.get("name") == "pool", "must delegate to the off-thread pool seam"
    assert "ok" not in sync_called, (
        "must NOT call rpc.stop_pool synchronously on the UI thread"
    )
