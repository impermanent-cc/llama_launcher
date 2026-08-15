import json
from dataclasses import asdict
from pathlib import Path

from llama_launcher.core.spec import slugify

CAP = 5


def history_path(base_dir, profile_name) -> Path:
    return Path(base_dir) / "benchmarks" / f"{slugify(profile_name)}.json"


def load(base_dir, profile_name) -> list:
    try:
        data = json.loads(history_path(base_dir, profile_name).read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def append(base_dir, profile_name, run) -> list:
    runs = load(base_dir, profile_name)
    runs.append(asdict(run))
    runs = runs[-CAP:]
    path = history_path(base_dir, profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=2))
    return runs


def clear(base_dir, profile_name) -> None:
    """Delete this profile's on-disk benchmark history. Idempotent."""
    history_path(base_dir, profile_name).unlink(missing_ok=True)


def _pct(new, old):
    return None if not old else (new - old) / old * 100.0


def delta(new: dict, old: dict) -> dict:
    old_by = {r["target_size"]: r for r in old.get("rows", [])}
    shared = []
    for r in new.get("rows", []):
        o = old_by.get(r["target_size"])
        if o is None:
            continue
        shared.append({"size": r["target_size"],
                       "pp_pct": _pct(r["pp_tok_s"], o["pp_tok_s"]),
                       "gen_pct": _pct(r["gen_tok_s"], o["gen_tok_s"])})
    new_sizes = {r["target_size"] for r in new.get("rows", [])}
    old_sizes = {r["target_size"] for r in old.get("rows", [])}
    return {"shared": shared, "sizes_differ": new_sizes != old_sizes}
