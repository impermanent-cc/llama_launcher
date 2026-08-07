from dataclasses import dataclass

from .router_preset import convert_raw_args
from .spec import Profile, member_model_id
from .settings_catalog import CATALOG

# Ports Odysseus scans when discovering local model servers
# (src/model_discovery.py). A router outside these ranges is reachable but will
# not be found automatically.
ODYSSEUS_SCAN_PORTS: frozenset = frozenset(range(8000, 8021)) | {8080, 1234, 11434, 11435}

# Addresses that keep a published port on this machine. Anything else is
# reachable from the network. Kept deliberately narrow: an unlisted address is
# treated as exposed, which is the safe direction to be wrong in.
LOOPBACK_HOSTS: frozenset = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]",
})
_LOOPBACK_HOSTS = LOOPBACK_HOSTS   # retained: referenced by the router rules below


def dial_host(bind_host: str) -> str:
    """The address to CONNECT to for a server published on `bind_host`.

    0.0.0.0 (and ::) are bind wildcards, not destinations — dialing them is a
    bug, so they map to loopback.
    """
    return "127.0.0.1" if bind_host in ("0.0.0.0", "::", "[::]", "") else bind_host


@dataclass
class Issue:
    level: str   # "error" | "warning"
    message: str


def _under_any_mount(path: str, profile: Profile) -> bool:
    return any(path.startswith(m.container.rstrip("/") + "/") or path == m.container
               for m in profile.mounts)


def validate(profile: Profile, running_ports: tuple = (),
             binary_found: bool = True, members: tuple = (),
             api_key_present: bool = False) -> list[Issue]:
    issues: list[Issue] = []

    if not binary_found:
        issues.append(Issue("error",
                            f"Runtime '{profile.runtime.binary}' not found on PATH."))

    for m in profile.mounts:
        if bool(m.host) != bool(m.container):
            issues.append(Issue("error",
                                "Mount row is incomplete (host and container both required)."))

    # Exposure applies to BOTH modes: Runtime.bind_host drives the publish
    # address for every launch, so a single-model server bound past loopback
    # with no key is just as open as a router would be. In server mode the key
    # is the catalog setting; in router mode it comes from the key file.
    if profile.runtime.bind_host not in LOOPBACK_HOSTS:
        has_key = api_key_present if profile.mode == "router" \
            else bool(profile.settings.get("api-key"))
        if not has_key:
            issues.append(Issue(
                "error",
                f"Binding to {profile.runtime.bind_host} without an API key would expose an "
                f"unauthenticated server. Generate a key first."))

    if profile.mode == "router":
        issues += _validate_router(profile, members, api_key_present)
    else:
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

    # A draft model is inert unless a speculation strategy is selected: llama.cpp
    # defaults --spec-type to 'none', so the draft is loaded (costing VRAM) and
    # never used. Warn (don't block).
    if profile.draft_model and profile.settings.get("spec-type", "none") in ("none", "", None):
        issues.append(Issue("warning",
                            "A draft model is selected but spec-type is 'none', so the draft "
                            "model is loaded and never used. Set spec-type (e.g. draft-simple) "
                            "to enable speculative decoding, or clear the draft model."))

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


def _validate_router(profile: Profile, members: tuple,
                     api_key_present: bool) -> list[Issue]:
    issues: list[Issue] = []

    if not members:
        issues.append(Issue("error",
                            "A router needs at least one model; add member profiles."))

    seen_ids: dict = {}
    for member, member_profile in members:
        model_id = member_model_id(member)
        if model_id in seen_ids:
            issues.append(Issue(
                "error",
                f"Two members share the model id '{model_id}' "
                f"({seen_ids[model_id]} and {member_profile.name}); ids must be unique "
                f"because harnesses use them to route requests."))
        else:
            seen_ids[model_id] = member_profile.name

        if not member_profile.model:
            issues.append(Issue("error",
                                f"Member '{member_profile.name}' has no model selected."))
        elif not _under_any_mount(member_profile.model, profile):
            issues.append(Issue(
                "error",
                f"Member '{member_profile.name}' model path is not under any mount on "
                f"this router; the router container can't see it."))

        if len(member_profile.loras) > 1:
            issues.append(Issue(
                "warning",
                f"Member '{member_profile.name}' has more than one LoRA; a preset INI "
                f"can only express the first."))

        _pairs, problems = convert_raw_args(member_profile.raw_args)
        if problems:
            issues.append(Issue(
                "warning",
                f"Member '{member_profile.name}' raw args can't all be expressed in a "
                f"preset and will be dropped: {'; '.join(problems)}"))

    # NB: the non-loopback-without-a-key error is raised in validate() for both
    # modes, so it is deliberately not repeated here.

    models_max = profile.settings.get("models-max", CATALOG["models-max"].default)
    if isinstance(models_max, int) and models_max > 1:
        issues.append(Issue(
            "warning",
            f"models-max is {models_max}: the router may hold that many models resident "
            f"at once, which can exceed VRAM. Use 1 unless the members are small."))

    port = profile.settings.get("port", 8080)
    if port not in ODYSSEUS_SCAN_PORTS:
        issues.append(Issue(
            "warning",
            f"Port {port} is outside the ranges Odysseus scans when it discovers model "
            f"servers (8000-8020, 8080, 1234, 11434, 11435), so it won't be found "
            f"automatically."))

    # Default to the CATALOG default, not "": cors-origins defaults to "*", so
    # reading absence as "" made the most dangerous configuration unwarnable.
    origins = str(profile.settings.get("cors-origins", CATALOG["cors-origins"].default) or "")
    uses_agent_tools = bool(profile.settings.get("tools")
                            or profile.settings.get("agent")
                            or profile.settings.get("mcp-servers-config")
                            or profile.settings.get("mcp-servers-json"))
    if uses_agent_tools and origins and origins not in ("localhost", "*"):
        issues.append(Issue(
            "warning",
            "--tools/--agent/MCP clamp CORS origins to localhost, so the origin set here "
            "will be overridden."))

    if (profile.runtime.bind_host not in _LOOPBACK_HOSTS
            and origins == "*"
            and not profile.settings.get("cors-credentials")):
        issues.append(Issue(
            "warning",
            "CORS origins '*' with credentials enabled echoes the Origin header back and "
            "always allows credentials, on a port exposed beyond loopback."))

    return issues
