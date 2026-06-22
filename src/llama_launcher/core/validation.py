from dataclasses import dataclass

from .spec import Profile


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

    return issues
