import os
import subprocess
import datetime
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox

from llama_launcher.core.spec import Profile
from llama_launcher.core.validation import validate, dial_host
from llama_launcher.store.profiles import profile_to_dict, load_config, save_config
from llama_launcher.services import runtime, gpu, metrics
from llama_launcher.services import api_key as api_key_store
from llama_launcher.core import report as report_mod
from llama_launcher.ui.dialogs.report_dialog import ReportDialog


class ReportController:
    """Owns report/export/web-ui behavior. Owns no workers -- no drain().

    Widgets stay on the window (built by MainWindow -- `monitor_panel`,
    `status_label`, etc.); this controller only owns behavior. Members this
    controller itself owns (e.g. `gather_report_data`, `_metrics_report_text`,
    `_save_report`, `on_generate_report`, `_on_export_sh`, `open_web_ui`,
    `export_sh`) are called directly as `self.<method>(...)`; widgets and
    methods owned by other panels/controllers go through
    `self.window._<owner>.<x>` (e.g. `self.window._configure_panel.
    current_profile`, `self.window._monitor._poll_api_key`).

    `base_dir` is looked up via a deferred
    `from llama_launcher.ui.main_window import base_dir` inside the methods
    that use it (not imported at module scope here), because
    tests/ui/test_report.py monkeypatches it as `mw.base_dir` -- a rebind of
    the name on main_window's module namespace. A fresh per-call import
    resolves that rebind; a module-scope `from ... import base_dir` here
    would bind a stale copy that the monkeypatch can't reach.

    `ReportDialog` is imported at module scope from its canonical home
    (`llama_launcher.ui.dialogs.report_dialog`); tests patch it via
    `monkeypatch.setattr("llama_launcher.ui.controllers.report_controller.ReportDialog", ...)`.
    """

    def __init__(self, window):
        self.window = window

    def _on_export_sh(self):
        path, _ = QFileDialog.getSaveFileName(self.window, "Export shell script", "run.sh",
                                              "Shell scripts (*.sh);;All files (*)")
        if path:
            self.export_sh(path)

    def open_web_ui(self):
        p = self.window._configure_panel.current_profile()
        port = p.settings.get("port", 8080)
        try:
            subprocess.Popen(["xdg-open", f"http://{dial_host(p.runtime.bind_host)}:{port}"],
                             start_new_session=True)
        except OSError:
            QMessageBox.warning(self.window, "Open Web UI", "Could not open browser (xdg-open not found).")

    def export_sh(self, path: str):
        cmd = " ".join(self.window._configure_panel.build_current_command())
        Path(path).write_text(f"#!/usr/bin/env bash\n{cmd}\n")
        os.chmod(path, 0o755)

    def gather_report_data(self) -> dict:
        import platform, json as _json
        p = self.window._configure_panel.current_profile()
        cmd = " ".join(self.window._configure_panel.build_current_command(p))
        # Pass the router context, or the report claims a healthy router has no
        # members and is exposed without a key -- in the one artifact users
        # paste when asking for help.
        issues = validate(p, binary_found=runtime.binary_available(p.runtime.binary),
                          members=self.window._configure_panel.member_pairs(),
                          api_key_present=bool(
                              api_key_store.resolve_api_key(self.window.router_base_dir(), p))
                          if p.mode == "router" else False)
        gpus = gpu.query_gpus()
        gpu_txt = "\n".join(f"{g.name}: {g.mem_used_mib}/{g.mem_total_mib} MiB, "
                            f"util {g.util_pct}%, {g.temp_c}C" for g in gpus) or "(no nvidia-smi)"
        runtime_txt = (f"binary={p.runtime.binary} gpu_mode={p.runtime.gpu_mode}\n"
                       f"rootless={runtime.is_rootless(p.runtime.binary)}\n"
                       f"{gpu_txt}\nOS={platform.platform()}")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "generated_at": ts,
            "command": report_mod.redact_secrets(cmd),
            "profile": report_mod.redact_secrets(_json.dumps(profile_to_dict(p), indent=2)),
            "validation": [f"[{i.level}] {i.message}" for i in issues],
            "status_history": [self.window.status_label.text()],
            "runtime": runtime_txt,
            "metrics": self._metrics_report_text(p),
            "image": p.image,
            "logs": report_mod.redact_secrets(self.window.monitor_panel.log_view.toPlainText()[-4000:]),
        }

    def _metrics_report_text(self, p: Profile) -> str:
        """Snapshot of the live /metrics endpoint for the diagnostic report.

        Returns a note (and makes no network call) when --metrics is off, so the
        report explains why throughput is missing instead of silently omitting it.
        """
        from llama_launcher.services.metrics import kv_ratio
        port = p.settings.get("port", 8080)
        if not p.settings.get("metrics"):
            return ("(--metrics not enabled in this profile — turn it on and relaunch "
                    "to capture tok/s and KV-cache usage here)")
        # Mirror collect_monitor_data's host/key/scope derivation: /metrics needs
        # the API key, and on a router it is per-model (?model=id) reached via the
        # router host. Without these the report's fetch 401'd (or returned nothing)
        # and always printed the "no metrics returned" note for routers.
        host = dial_host(p.runtime.bind_host)
        key = self.window._monitor._poll_api_key(p)
        model_scope = None
        if p.mode == "router":
            host = self.window._monitor._router_host(p)
            model_scope = self.window._monitor._router_pollable_model()
        m = metrics.fetch_metrics(port, model=model_scope, api_key=key, host=host)
        slots = metrics.fetch_slots(port, model=model_scope, api_key=key, host=host)
        if not m and not slots:
            scope = " (no model currently loaded on the router)" if (
                p.mode == "router" and model_scope is None) else ""
            return (f"(no metrics returned from http://{host}:{port}/metrics{scope} — "
                    "generate the report while the server is running with --metrics)")
        lines = []
        gen = m.get("llamacpp:predicted_tokens_seconds")
        if gen is not None:
            lines.append(f"generation: {gen:.2f} tok/s")
        prompt = m.get("llamacpp:prompt_tokens_seconds")
        if prompt is not None:
            lines.append(f"prompt: {prompt:.2f} tok/s")
        kv = kv_ratio(m, slots)
        if kv is not None:
            lines.append(f"KV cache usage: {kv * 100:.0f}%")
        if m:
            if lines:
                lines.append("")
            lines += [f"{k} {v:g}" for k, v in sorted(m.items())]
        return "\n".join(lines)

    def _save_report(self, md: str, ts: str | None = None) -> Path:
        from llama_launcher.ui.main_window import base_dir
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        reports_dir = base_dir() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out = reports_dir / f"llama-launcher-report-{ts}.md"
        out.write_text(md)
        return out

    def on_generate_report(self):
        from llama_launcher.ui.main_window import base_dir
        cfg = load_config(base_dir())
        initial = cfg.get("report_sections", {s: True for s in report_mod.REPORT_SECTIONS})
        dlg = ReportDialog(initial, self.window)
        if not dlg.exec():
            return
        sections = dlg.selected_sections()
        cfg["report_sections"] = sections
        save_config(cfg, base_dir())
        data = self.gather_report_data()
        md = report_mod.build_report(data, sections)
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(md)
        saved = self._save_report(md, data.get("generated_at"))
        QMessageBox.information(self.window, "Report saved", f"Report saved to:\n{saved}")
