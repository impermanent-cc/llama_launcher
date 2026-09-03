from unittest.mock import patch

from llama_launcher.ui.controllers import monitor_controller as mc
from llama_launcher.ui.widgets.stat_card import StatCard


def _target(tmp_path):
    return {
        "binary": "podman",
        "base_dir": str(tmp_path),
        "router_base_dir": str(tmp_path),
        "nodes": [
            {
                "name": "local",
                "connection": "",
                "host": "127.0.0.1",
                "binary": "podman",
                "enabled": True,
            },
            {
                "name": "box-b",
                "connection": "box-b",
                "host": "192.168.1.11",
                "binary": "podman",
                "enabled": True,
            },
        ],
    }


def test_gather_tags_rows_by_node_and_survives_unreachable(tmp_path):
    def fake_list(binary, connection=""):
        if connection == "box-b":
            raise OSError("unreachable")  # must not blow up the gather
        return [{"name": "llama-p", "running": True, "profile": "p", "mode": "server"}]

    with (
        patch.object(mc.runtime, "list_launcher_containers", side_effect=fake_list),
        patch.object(mc, "list_profiles", return_value=[]),
        patch.object(mc.native, "list_native_instances", return_value=[]),
    ):
        data = mc.build_instances_data(_target(tmp_path))

    nodes = {i.node for i in data["instances"]}
    assert "local" in nodes  # local rows present
    assert "box-b" not in nodes  # unreachable node dropped, no crash


def test_disabled_node_is_skipped(tmp_path):
    target = _target(tmp_path)
    target["nodes"][1]["enabled"] = False
    calls = []

    def fake_list(binary, connection=""):
        calls.append(connection)
        return [{"name": "llama-p", "running": True, "profile": "p", "mode": "server"}]

    with (
        patch.object(mc.runtime, "list_launcher_containers", side_effect=fake_list),
        patch.object(mc, "list_profiles", return_value=[]),
        patch.object(mc.native, "list_native_instances", return_value=[]),
    ):
        mc.build_instances_data(target)

    assert calls == [""]  # box-b never queried


def test_row_carries_node_and_local_rows_use_node_local(tmp_path):
    with (
        patch.object(
            mc.runtime,
            "list_launcher_containers",
            return_value=[
                {"name": "llama-p", "running": True, "profile": "p", "mode": "server"}
            ],
        ),
        patch.object(mc, "list_profiles", return_value=[]),
        patch.object(mc.native, "list_native_instances", return_value=[]),
    ):
        data = mc.build_instances_data(
            {
                "binary": "podman",
                "base_dir": str(tmp_path),
                "router_base_dir": str(tmp_path),
                "nodes": [
                    {
                        "name": "local",
                        "connection": "",
                        "host": "127.0.0.1",
                        "binary": "podman",
                        "enabled": True,
                    }
                ],
            }
        )
    assert data["rows"][0]["node"] == "local"


def test_remote_row_renders_node_name_on_card(qtbot):
    c = StatCard("llama-p")
    qtbot.addWidget(c)
    c.update_row(
        {
            "profile": "p",
            "port": 8080,
            "health": "ready",
            "tok_s": 12.0,
            "kv_pct": 0.1,
            "embeddings": False,
            "reranking": False,
            "mode": "server",
            "running": True,
            "node": "box-b",
        }
    )
    assert "box-b" in c.title_text()


def test_local_row_card_title_has_no_node_suffix(qtbot):
    c = StatCard("llama-p")
    qtbot.addWidget(c)
    c.update_row(
        {
            "profile": "p",
            "port": 8080,
            "health": "ready",
            "tok_s": 12.0,
            "kv_pct": 0.1,
            "embeddings": False,
            "reranking": False,
            "mode": "server",
            "running": True,
            "node": "local",
        }
    )
    assert c.title_text() == "p  :8080"
