"""Pure host<->container path translation. No Qt / IO imports."""


def host_to_container(host_path: str, mounts) -> str | None:
    """Return the container-side path for a host file living under a mount.

    Each mount is matched by its host prefix (trailing '/' stripped):
      - exact match -> mount.container
      - host_path under host + '/' -> container (no trailing '/') + '/' + relative
    Mounts with an empty host or container are skipped.
    Returns None if host_path is under none of the mounts.
    """
    for m in mounts:
        host = (m.host or "").rstrip("/")
        container = m.container or ""
        if not host or not container:
            continue
        if host_path == host:
            return container
        prefix = host + "/"
        if host_path.startswith(prefix):
            rel = host_path[len(prefix):]
            return container.rstrip("/") + "/" + rel
    return None
