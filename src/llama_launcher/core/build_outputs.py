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
    rows = []
    for output in outputs:
        if output.kind == "tag":
            # Tag kind: check if in images dict
            if output.identifier in images:
                img = images[output.identifier]
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
    # Collect all identifiers from tag-kind outputs
    claimed_tags = {o.identifier for o in outputs if o.kind == "tag"}

    # Find tags with custom repo and not claimed
    untracked = []
    for tag in images.keys():
        # Extract repo part (before the colon)
        repo_part = tag.split(":")[0] if ":" in tag else tag
        if repo_part.endswith("-custom") and tag not in claimed_tags:
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
