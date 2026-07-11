from dataclasses import dataclass

from .spec import Profile
from .settings_catalog import CATALOG


@dataclass
class Issue:
    level: str   # "error" | "warning"
    message: str


def _under_any_mount(path: str, profile: Profile) -> bool:
    return any(path.startswith(m.container.rstrip("/") + "/") or path == m.container
               for m in profile.mounts)


def validate(profile: Profile, running_ports: tuple = (),
             binary_found: bool = True) -> list[Issue]:
    issues: list[Issue] = []

    if not binary_found:
        issues.append(Issue("error",
                            f"Runtime '{profile.runtime.binary}' not found on PATH."))

    for m in profile.mounts:
        if bool(m.host) != bool(m.container):
            issues.append(Issue("error",
                                "Mount row is incomplete (host and container both required)."))

    if not profile.model:
        issues.append(Issue("error", "No model selected."))
    elif not _under_any_mount(profile.model, profile):
        issues.append(Issue("error",
                            "Model path is not under any mounted folder; the "
                            "container can't see it."))

    if profile.mmproj and not _under_any_mount(profile.mmproj, profile):
        issues.append(Issue("error", "mmproj path is not under any mount."))

    for lora in profile.loras:
        if not _under_any_mount(lora.path, profile):
            issues.append(Issue("error", f"LoRA path not under any mount: {lora.path}"))

    if profile.settings.get("tools"):
        for m in profile.mounts:
            if m.role == "model" and m.mode == "rw":
                issues.append(Issue("warning",
                                    "Tools are enabled and a model mount is writable; "
                                    "your weights are writable by the model."))
                break

    port = profile.settings.get("port", 8080)
    if port in running_ports:
        issues.append(Issue("warning",
                            f"Port {port} is already used by a running launcher container."))

    # MTP speculative decoding (--spec-type draft-mtp) has two known limitations
    # in llama.cpp: it ignores the multimodal projector and only supports a
    # single slot. Warn (don't block) — these run but silently lose the feature.
    if profile.settings.get("spec-type") == "draft-mtp":
        if profile.mmproj:
            issues.append(Issue("warning",
                                "MTP (--spec-type draft-mtp) doesn't support --mmproj; the "
                                "multimodal projector is likely ignored. Drop the mmproj for "
                                "a text-only MTP run, or use a non-MTP draft for vision."))
        parallel = profile.settings.get("parallel")
        if isinstance(parallel, int) and parallel > 1:
            issues.append(Issue("warning",
                                "MTP (--spec-type draft-mtp) doesn't support --parallel > 1; "
                                "set parallel = 1 (a single slot)."))

    # Embedding / reranking bad-combo warnings. A reranker needs all three of
    # --reranking, --pooling rank, and --embeddings; sampling is ignored here.
    if profile.settings.get("embeddings") or profile.settings.get("reranking"):
        if profile.settings.get("reranking"):
            if profile.settings.get("pooling") != "rank":
                issues.append(Issue("warning",
                                    "Reranking needs --pooling rank; other pooling types give "
                                    "near-zero scores. Set pooling = rank."))
            if not profile.settings.get("embeddings"):
                issues.append(Issue("warning",
                                    "Reranking needs --embeddings enabled (embedding "
                                    "extraction). Enable it."))
        changed = sorted(k for k, s in CATALOG.items()
                         if s.group == "Sampling" and k in profile.settings
                         and profile.settings[k] != s.default)
        if changed:
            issues.append(Issue("warning",
                                "Sampling parameters are ignored in embedding mode "
                                f"(changed: {', '.join(changed)})."))

    return issues
