from dataclasses import dataclass
from llama_launcher.core.build_spec import BuildOutput


@dataclass
class OutputRow:
    """Classification of a build output with its usage status."""
    output: BuildOutput | None
    identifier: str
    status: str  # "built", "missing", or "untracked"
    size: str = ""
    created: str = ""


def _normalize_tag(tag: str) -> str:
    """Rootless podman names locally built images ``localhost/<repo>:<tag>``
    while the registry stores the unqualified tag the user was told to build.
    Strip that prefix so the two spellings compare equal; podman itself
    resolves either spelling (rmi included)."""
    return tag[len("localhost/"):] if tag.startswith("localhost/") else tag


def classify_outputs(
    outputs: list[BuildOutput],
    images: dict[str, "ImageInfo"],
    binary_exists,
) -> list[OutputRow]:
    """Classify each BuildOutput as built or missing.

    Args:
        outputs: List of BuildOutput records.
        images: Dict mapping image tag to ImageInfo (with tag, size, created fields).
        binary_exists: Callable that returns True if a binary path exists.

    Returns:
        List of OutputRow with status in {"built", "missing"}.
    """
    by_normalized = {_normalize_tag(t): img for t, img in images.items()}
    rows = []
    for output in outputs:
        if output.kind == "tag":
            # localhost/-insensitive image lookup; only images carry metadata.
            img = by_normalized.get(_normalize_tag(output.identifier))
            rows.append(OutputRow(
                output=output,
                identifier=output.identifier,
                status="built" if img is not None else "missing",
                size=img.size if img is not None else "",
                created=img.created if img is not None else "",
            ))
        elif output.kind == "binary":
            rows.append(OutputRow(
                output=output,
                identifier=output.identifier,
                status="built" if binary_exists(output.identifier) else "missing",
            ))
    return rows


def untracked_custom_tags(
    images: dict[str, "ImageInfo"],
    outputs: list[BuildOutput],
) -> list[str]:
    """Find custom-repo tags that no BuildOutput claims.

    Args:
        images: Dict mapping image tag to ImageInfo.
        outputs: List of BuildOutput records.

    Returns:
        List of tag strings where repo part ends with "-custom" and no output
        has that identifier.
    """
    # Collect all identifiers from tag-kind outputs (localhost/-insensitive)
    claimed_tags = {_normalize_tag(o.identifier) for o in outputs if o.kind == "tag"}

    # Find tags with custom repo and not claimed. Report podman's OWN
    # spelling (prefix kept) so rmi on an untracked row works verbatim.
    untracked = []
    for tag in images.keys():
        # Extract repo part (before the trailing tag version)
        # Use rsplit to handle registry:port formats like registry:5000/x-custom:v1
        repo_part = tag.rsplit(":", 1)[0] if ":" in tag else tag
        if repo_part.endswith("-custom") and _normalize_tag(tag) not in claimed_tags:
            untracked.append(tag)

    return untracked


def extract_build_dir(path: str) -> str:
    """Extract the build directory from a path like /s/build-x/bin/llama-server.

    Returns the directory that contains the build (dirname(dirname(path))).
    Uses string manipulation to avoid filesystem imports. This is the single
    "which build dir owns this binary" rule: both the in-use guard below and
    the Build tab's delete path derive the directory through it, so the
    layout assumption can never diverge between the check and the rmtree.
    """
    # Split by / and take all parts except the last 2
    parts = path.rstrip("/").split("/")
    if len(parts) > 2:
        return "/".join(parts[:-2])
    return ""


def profiles_using(
    identifier: str,
    kind: str,
    profiles: list,
) -> list[str]:
    """Find profile names using the given identifier.

    Args:
        identifier: Image tag or binary path to search for.
        kind: "tag" or "binary".
        profiles: List of Profile objects.

    Returns:
        List of profile names using the identifier.
    """
    result = []
    # Tags compare localhost/-insensitively on BOTH sides: profiles can hold
    # podman's own `localhost/<tag>` spelling (use-in-profile on an untracked
    # row writes it verbatim) while the registry stores the unqualified tag.
    want_tag = _normalize_tag(identifier) if kind == "tag" else ""
    id_build_dir = extract_build_dir(identifier) if kind == "binary" else ""

    for profile in profiles:
        if kind == "tag":
            image = getattr(profile, "image", None)
            if image and _normalize_tag(image) == want_tag:
                result.append(profile.name)
            continue
        if kind != "binary":
            continue
        native_binary = getattr(getattr(profile, "runtime", None),
                                "native_binary", "")
        if not native_binary:
            continue
        # Direct match, or living inside the identifier's build-<slug> dir.
        if native_binary == identifier or (
                id_build_dir and extract_build_dir(native_binary) == id_build_dir):
            result.append(profile.name)

    return result
