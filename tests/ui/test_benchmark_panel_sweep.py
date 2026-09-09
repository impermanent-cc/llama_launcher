from llama_launcher.ui.panels.benchmark_panel import BenchmarkPanel


def _sweep(best=4):
    def pt(c, status, gen):
        return {
            "count": c,
            "status": status,
            "ready_seconds": 7.5,
            "error": "" if status == "ok" else "OOM",
            "rows": [
                {
                    "target_size": 128,
                    "prompt_n": 128,
                    "pp_tok_s": 800.0,
                    "gen_tok_s": gen + 1,
                    "total_s": 1.0,
                },
                {
                    "target_size": 2048,
                    "prompt_n": 2048,
                    "pp_tok_s": 700.0,
                    "gen_tok_s": gen,
                    "total_s": 2.0,
                },
            ]
            if status == "ok"
            else [],
            "measured": {
                "cards": [{"model": 10 * 2**30, "kv": 2**30, "compute": 2**29}],
                "ram": {"model": 2**30, "kv": 0, "compute": 0, "output": 2**29},
            }
            if status == "ok"
            else None,
            "estimated_cards": [12 * 2**30],
            "estimated_ram": 2 * 2**30,
        }

    return {
        "profile": "p",
        "knob": "n-cpu-ffn",
        "timestamp": "t",
        "bench_cfg": {},
        "points": [pt(0, "failed", 0), pt(2, "ok", 30.0), pt(best, "ok", 33.0)],
    }


def test_prefill_and_config(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.set_sweep_prefill("n-cpu-moe", 12, 20, 2)
    assert "n-cpu-moe" in panel.sweep_knob_label.text()
    cfg = panel.sweep_config()
    assert (cfg["start"], cfg["stop"], cfg["step"]) == (12, 20, 2)
    assert cfg["ready_timeout"] == 600


def test_run_click_emits_config_and_cancel_when_running(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.set_sweep_available(True)
    got = []
    panel.sweep_run_requested.connect(got.append)
    panel.sweep_run_btn.click()
    assert got and got[0]["step"] == 2
    panel.set_sweep_running(True)
    cancelled = []
    panel.sweep_cancel_requested.connect(lambda: cancelled.append(1))
    panel.sweep_run_btn.click()
    assert cancelled and panel.sweep_run_btn.text() == "Cancel sweep"


def test_show_sweep_marks_best_and_fills_columns(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep(_sweep(), n_cards=1)
    t = panel.sweep_table
    assert t.horizontalHeader().stretchLastSection() is True
    assert t.rowCount() == 3
    assert t.item(0, 1).text() == "failed" and "OOM" in t.item(0, 1).toolTip()
    assert t.item(2, 0).text().startswith("4") and "best" in t.item(2, 0).text()
    assert t.item(2, 5).text().startswith("11.5 / 12.0")
    assert t.item(2, 6).text().startswith("1.5 / 2.0")
    assert panel.sweep_apply_btn.isEnabled()
    got = []
    panel.sweep_apply_requested.connect(got.append)
    panel.sweep_apply_btn.click()
    assert got == [4]


def test_unavailable_reason_shows_and_disables(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.set_sweep_available(False, "Sweep needs a container profile.")
    assert not panel.sweep_run_btn.isEnabled()
    assert "container profile" in panel.sweep_status.text()


def test_set_sweep_point_updates_and_appends_rows(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    empty = {
        "profile": "p",
        "knob": "n-cpu-ffn",
        "timestamp": "t",
        "bench_cfg": {},
        "points": [],
    }
    panel.show_sweep(empty, n_cards=2)
    t = panel.sweep_table

    panel.set_sweep_point(
        {"count": 4, "status": "running", "rows": [], "measured": None}
    )
    assert t.rowCount() == 1
    assert t.item(0, 0).text() == "4"
    assert t.item(0, 3).text() == "-" and t.item(0, 4).text() == "-"
    assert not panel.sweep_apply_btn.isEnabled()

    panel.set_sweep_point(
        {
            "count": 4,
            "status": "ok",
            "ready_seconds": 5.0,
            "rows": [
                {
                    "target_size": 128,
                    "prompt_n": 128,
                    "pp_tok_s": 800.0,
                    "gen_tok_s": 40.0,
                    "total_s": 1.0,
                }
            ],
            "measured": {
                "cards": [{"model": 2**30, "kv": 0, "compute": 0}],
                "ram": {"model": 0, "kv": 0, "compute": 0, "output": 0},
            },
            "estimated_cards": [2**30],
            "estimated_ram": 0,
        }
    )
    assert t.rowCount() == 1
    assert t.item(0, 1).text() == "ok"

    panel.set_sweep_point(
        {"count": 6, "status": "running", "rows": [], "measured": None}
    )
    assert t.rowCount() == 2
    assert panel.sweep_apply_btn.isEnabled()


def test_show_sweep_tolerates_partial_points_without_raising(qtbot):
    """A point missing status/measured, and a row missing target_size, paint
    with dash placeholders instead of raising."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    sweep = {
        "profile": "p",
        "knob": "n-cpu-ffn",
        "timestamp": "t",
        "bench_cfg": {},
        "points": [{"count": 3, "rows": [{"pp_tok_s": 5.0}]}],
    }
    panel.show_sweep(sweep, n_cards=1)
    t = panel.sweep_table
    assert t.rowCount() == 1
    assert t.item(0, 1).text() == ""
    assert t.item(0, 4).text() == "-"
    assert t.item(0, 5).text() == "- / -"


def test_reset_clears_the_sweep_table_and_apply(qtbot):
    """A reset drops the previous profile's sweep rows, its best mark and its
    enabled Apply."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep(_sweep(), n_cards=1)
    assert panel.sweep_table.rowCount() == 3 and panel.sweep_apply_btn.isEnabled()

    panel.reset()

    assert panel.sweep_table.rowCount() == 0
    assert not panel.sweep_apply_btn.isEnabled()
    assert panel._sweep_points == [] and panel._sweep_best_count is None


def test_show_sweep_labels_the_stored_timestamp_and_clear_removes_it(qtbot):
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep({**_sweep(), "timestamp": "2026-09-07T12:00:00"}, 1)
    assert "2026-09-07T12:00:00" in panel.sweep_stamp.text()
    panel.clear_sweep()
    assert panel.sweep_stamp.text() == ""


def test_show_sweep_with_no_timestamp_leaves_the_stamp_blank(qtbot):
    """The empty table shown while a fresh sweep starts carries no
    timestamp, so the stamp stays blank instead of a bare 'stored'."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep({"points": []}, 1)
    assert panel.sweep_stamp.text() == ""


def test_show_sweep_with_stored_false_leaves_the_stamp_blank(qtbot):
    """A cancelled sweep is shown but never written to disk, so its own
    timestamp must not be labelled as a stored one."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep({**_sweep(), "timestamp": "2026-09-07T12:00:00"}, 1, stored=False)
    assert panel.sweep_stamp.text() == ""


def test_bad_prompt_sizes_do_not_emit_a_benchmark_run(qtbot):
    """A prompt-size token that is not an int refuses the run silently: the
    panel emits nothing rather than a run with a bad size."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.set_benchmark_available(True)
    panel.bench_sizes.setText("128, x")
    got = []
    panel.benchmark_run_requested.connect(got.append)
    panel.bench_run_btn.click()
    assert got == []


def test_the_row_shown_is_the_last_of_equal_prompt_sizes(qtbot):
    """The table reads the same row best_point ranks on: tied prompt sizes
    resolve to the last one."""
    panel = BenchmarkPanel()
    qtbot.addWidget(panel)
    panel.show_sweep(
        {
            "points": [
                {
                    "count": 0,
                    "status": "ok",
                    "rows": [
                        {"target_size": 2048, "pp_tok_s": 700.0, "gen_tok_s": 20.0},
                        {"target_size": 2048, "pp_tok_s": 600.0, "gen_tok_s": 10.0},
                    ],
                    "measured": None,
                }
            ]
        },
        n_cards=0,
    )
    assert panel.sweep_table.item(0, 3).text() == "600.0"
    assert panel.sweep_table.item(0, 4).text() == "10.0"
