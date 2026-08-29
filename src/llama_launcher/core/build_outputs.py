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
            # Tag kind: check if in images dict (localhost/-insensitive)
            if _normalize_tag(output.identifier) in by_normalized:
                img = by_normalized[_normalize_tag(output.identifier)]
                rows.append(OutputRow(
                    output=output,
                    identifier=output.identifier,
                    status="built",
                    size=getattr(img, "size", ""),
                    created=getattr(img, "created", ""),
                ))
            else:
                rows.append(OutputRow(
                    output=output,
                    identifier=output.identifier,
                    status="missing",
                ))
        elif output.kind == "binary":
            # Binary kind: use binary_exists callable
            if binary_exists(output.identifier):
                rows.append(OutputRow(
                    output=output,
                    identifier=output.identifier,
                    status="built",
                ))
            else:
                rows.append(OutputRow(
                    output=output,
                    identifier=output.identifier,
                    status="missing",
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


def _extract_build_dir(path: str) -> str:
    """Extract the build directory from a path like /s/build-x/bin/llama-server.

    Returns the directory that contains the build (dirname(dirname(path))).
    Uses string manipulation to avoid filesystem imports.
    """
    # Split by / and take all parts except the last 2
    parts = path.split("/")
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

    for profile in profiles:
        if kind == "tag":
            # Check if profile.image matches identifier
            if getattr(profile, "image", None) == identifier:
                result.append(profile.name)
        elif kind == "binary":
            # Check if profile.runtime.native_binary equals identifier
            # or is inside its build-<slug> directory
            runtime = getattr(profile, "runtime", None)
            if runtime:
                native_binary = getattr(runtime, "native_binary", "")
                if native_binary:
                    # Direct match
                    if native_binary == identifier:
                        result.append(profile.name)
                    else:
                        # Check if native_binary is inside identifier's build dir
                        # Extract build directories and compare
                        id_build_dir = _extract_build_dir(identifier)
                        bin_build_dir = _extract_build_dir(native_binary)
                        if id_build_dir and bin_build_dir and id_build_dir == bin_build_dir:
                            result.append(profile.name)

    return result
