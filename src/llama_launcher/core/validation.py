import ipaddress
import re
from dataclasses import dataclass

from .command_builder import (raw_arg_warnings, dangerous_run_args,
                              run_args_expose, is_sensitive_host_path)
from .router_preset import convert_raw_args
from .spec import DEFAULT_PORT, Profile, member_model_id, profile_port
from .settings_catalog import CATALOG

# A model id becomes an INI section header ([id]) in a router preset and is sent
# by harnesses in the request "model" field, so keep it to a safe charset -- a
# newline would inject arbitrary preset keys into other sections.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")


def _bind_host_is_addressish(bind_host: str) -> bool:
    """True if bind_host is an IP literal, a bind wildcard, or an accepted
    loopback NAME. A free-text hostname is refused: bind_host doubles as the
    Monitor's dial target, so a hostname there would send the server's bearer
    token to an arbitrary host."""
    if bind_host in LOOPBACK_HOSTS or bind_host in ("0.0.0.0", "::", "[::]"):
        return True
    host = bind_host[1:-1] if bind_host.startswith("[") and bind_host.endswith("]") else bind_host
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False

# Addresses that keep a published port on this machine. Anything else is
# reachable from the network. Kept deliberately narrow: an unlisted address is
# treated as exposed, which is the safe direction to be wrong in.
LOOPBACK_HOSTS: frozenset = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]",
})
_LOOPBACK_HOSTS = LOOPBACK_HOSTS   # retained: referenced by the router rules below


def dial_host(bind_host: str) -> str:
    """The address to CONNECT to for a server published on `bind_host`.

    0.0.0.0 (and ::) are bind wildcards, not destinations; dialing them is a
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
             api_key_present: bool = False, native_binary_ok: bool = True,
             image_present: bool = True, worker_image_present: dict = None,
             worker_free_mb: dict = None) -> list[Issue]:
    """`native_binary_ok` mirrors `binary_found`: core stays I/O-free, so the
    caller stats `profile.runtime.native_binary` (e.g. via
    services.native.native_binary_available) and passes the result in.

    `worker_image_present`/`worker_free_mb` are the RPC-pool equivalents:
    dependency-injected {node: value} maps for `_validate_rpc`, populated by a
    caller that has actually probed the worker nodes."""
    issues: list[Issue] = []
    is_native = profile.runtime.launch_mode == "native"

    if is_native:
        if profile.mode == "router":
            issues.append(Issue("error",
                "Native launch does not support router mode in this version; "
                "use a container runtime for router profiles."))
        nb = profile.runtime.native_binary
        if not nb:
            issues.append(Issue("error", "Native mode needs a llama-server binary path."))
        elif not native_binary_ok:
            issues.append(Issue("error",
                f"Native binary not found or not executable: {nb}"))

    if not is_native and not binary_found:
        issues.append(Issue("error",
                            f"Runtime '{profile.runtime.binary}' not found on PATH."))

    is_ik = profile.runtime.engine == "ik_llama.cpp"
    img = "" if is_native else profile.image.lower()
    looks_ik = "ik-llama" in img or "ik_llama" in img

    # ik_llama.cpp has no router: its llama-server carries no --models-preset (or
    # any preset/router option at all), so a router launch dies immediately on
    # "unknown argument". Verified against ik-llama-cpp:cu12-server. Keyed on
    # the image as well as the engine field, because the engine defaults to
    # llama.cpp and nothing sets it from the image.
    if profile.mode == "router" and (is_ik or looks_ik):
        issues.append(Issue("error",
            "ik_llama.cpp has no router mode (no --models-preset). Use the llama.cpp "
            "engine and a llama.cpp image for router profiles."))

    if not is_native:
        if is_ik and img and not looks_ik:
            issues.append(Issue(
                "warning",
                "Engine is ik_llama.cpp but the image doesn't look like an ik build "
                "(no 'ik-llama'/'ik_llama' in the ref); ik-only flags may be rejected. "
                "Use an ik-llama-cpp image."))
        elif not is_ik and looks_ik and profile.mode != "router":
            issues.append(Issue(
                "warning",
                "Engine is llama.cpp but the image looks like an ik_llama.cpp build; "
                "switch the Engine to ik_llama.cpp to reach its flags."))

        for m in profile.mounts:
            if bool(m.host) != bool(m.container):
                issues.append(Issue("error",
                                    "Mount row is incomplete (host and container both required)."))
            if m.host and is_sensitive_host_path(m.host):
                issues.append(Issue(
                    "warning",
                    f"Mount source {m.host!r} is a sensitive host path; the container "
                    f"gets {'write ' if m.mode == 'rw' else ''}access to it. Mount only "
                    f"the model directory you need."))

        if profile.image and not image_present:
            issues.append(Issue(
                "warning",
                f"Image {profile.image!r} is not present on the selected node; "
                f"pull it (or build/copy it there) before launching."))

    # Untrusted-profile screening: extra_run_args is spliced verbatim into the
    # `podman run` argv, so a shared profile.json could otherwise break out of
    # the container (host mounts, --privileged, --entrypoint) on one Launch.
    if not is_native:
        bad = dangerous_run_args(profile.runtime.extra_run_args)
        if bad:
            issues.append(Issue(
                "error",
                "Extra podman/docker run args request host-level access "
                f"({', '.join(bad)}); refusing to launch. Remove them if you did "
                f"not intend to grant the container access to the host."))

    # bind_host is free text on a loaded profile; it must be an address, not a
    # hostname (which the Monitor would then dial WITH the api key attached).
    if not _bind_host_is_addressish(profile.runtime.bind_host):
        issues.append(Issue(
            "error",
            f"Bind host {profile.runtime.bind_host!r} is not an IP address or a "
            f"recognized loopback name; set an address (127.0.0.1 or 0.0.0.0)."))

    # Exposure applies to BOTH modes: Runtime.bind_host drives the publish
    # address for every launch, so a single-model server bound past loopback
    # with no key is just as open as a router would be. In server mode the key
    # is the catalog setting; in router mode it comes from the key file.
    # extra_run_args (--network host / extra -p) can defeat the bind restriction,
    # so treat the profile as exposed then too.
    exposed = (profile.runtime.bind_host not in LOOPBACK_HOSTS
               or (not is_native and run_args_expose(profile.runtime.extra_run_args)))
    if exposed:
        # Match the renderer's blank-drop: a whitespace-only key is dropped from
        # argv, so it is NOT real authentication (the "blank key" exposure hole).
        has_key = api_key_present if profile.mode == "router" \
            else bool(str(profile.settings.get("api-key", "")).strip())
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
        elif not is_native and not _under_any_mount(profile.model, profile):
            issues.append(Issue("error",
                                "Model path is not under any mounted folder; the "
                                "container can't see it."))

        if (profile.mmproj and not is_native
                and not _under_any_mount(profile.mmproj, profile)):
            issues.append(Issue("error", "mmproj path is not under any mount."))

        for lora in profile.loras:
            if not is_native and not _under_any_mount(lora.path, profile):
                issues.append(Issue("error", f"LoRA path not under any mount: {lora.path}"))

    if profile.settings.get("tools"):
        for m in profile.mounts:
            if m.role == "model" and m.mode == "rw":
                issues.append(Issue("warning",
                                    "Tools are enabled and a model mount is writable; "
                                    "your weights are writable by the model."))
                break

    port = profile_port(profile)
    if port in running_ports:
        issues.append(Issue("warning",
                            f"Port {port} is already used by a running launcher container."))

    # Single-server-only launch warnings, router-gated ONCE for the whole
    # stretch: a router profile keeps the form's leftover draft/model fields
    # and member-level settings (so a Save in router mode is not destructive),
    # but the router process itself loads no model -- every settings-derived
    # single-server warning below would be a false alarm there. New warnings
    # of this class belong INSIDE this block.
    if profile.mode != "router":
        # MTP speculative decoding (--spec-type draft-mtp) has two known
        # limitations in llama.cpp: it ignores the multimodal projector and
        # only supports a single slot. Warn (don't block); these run but
        # silently lose the feature.
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

        # A draft model is inert unless a speculation strategy is selected:
        # llama.cpp defaults --spec-type to 'none', so the draft is loaded
        # (costing VRAM) and never used. Warn (don't block).
        if profile.draft_model \
                and profile.settings.get("spec-type", "none") in ("none", "", None):
            issues.append(Issue("warning",
                                "A draft model is selected but spec-type is 'none', so the draft "
                                "model is loaded and never used. Set spec-type (e.g. draft-simple) "
                                "to enable speculative decoding, or clear the draft model."))

        if profile.runtime.engine == "ik_llama.cpp" \
                and profile.settings.get("run-time-repack"):
            msg = ("Run-time repack (--run-time-repack) disables mmap and increases load "
                   "time and RAM.")
            if profile.settings.get("load-mode", "mmap") == "mmap":
                msg += " Your load-mode is mmap, which it overrides."
            issues.append(Issue("warning", msg))

        # Embedding / reranking bad-combo warnings. A reranker needs all three
        # of --reranking, --pooling rank, and --embeddings; sampling is
        # ignored here.
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

    if profile.runtime.launch_mode == "rpc":
        issues += _validate_rpc(profile, worker_image_present or {}, worker_free_mb or {})

    for w in raw_arg_warnings(profile):
        issues.append(Issue("warning", w))

    return issues


_CENTRALIZING = ("cpu-moe", "n-cpu-moe", "no-kv-offload", "override-tensor")


def _validate_rpc(profile: Profile, worker_image_present: dict,
                  worker_free_mb: dict) -> list[Issue]:
    issues: list[Issue] = []
    if profile.mode == "router":
        issues.append(Issue("error", "RPC pooling does not support router mode."))
    workers = profile.runtime.rpc_workers
    if not workers:
        issues.append(Issue("error", "An RPC pool needs at least one worker."))
    for w in workers:
        if worker_image_present.get(w.node) is False:
            issues.append(Issue("error",
                f"RPC image {profile.image!r} is not present on node '{w.node}'; "
                f"build or copy the GGML_RPC image there before launching."))
        free = worker_free_mb.get(w.node)
        if free is not None and w.mem_mb and w.mem_mb > free:
            issues.append(Issue("warning",
                f"Worker on '{w.node}' donates {w.mem_mb} MB but only {free} MB is "
                f"free; more than the node has risks the head crashing mid-upload."))
    if any(profile.settings.get(k) for k in _CENTRALIZING):
        issues.append(Issue("warning",
            "A memory-centralizing flag (cpu-moe/n-cpu-moe/no-kv-offload/"
            "override-tensor) centralizes on the head; prefer -ngl spread across "
            "the pool for RPC."))
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
        if not _MODEL_ID_RE.match(model_id):
            issues.append(Issue(
                "error",
                f"Member '{member_profile.name}' has an invalid model id "
                f"{model_id!r}: use only letters, digits, and . _ : / - (it becomes "
                f"a preset section header and a routing key)."))
        if model_id in seen_ids:
            issues.append(Issue(
                "error",
                f"Two members share the model id '{model_id}' "
                f"({seen_ids[model_id]} and {member_profile.name}); ids must be unique "
                f"because harnesses use them to route requests."))
        else:
            seen_ids[model_id] = member_profile.name

        # The preset renders each member's settings for the MEMBER's engine,
        # but every member instance is spawned by the router's own llama-server
        # (mainline, the only engine with a router). An ik-tagged member would
        # write ik-only keys into the preset and the child would die on
        # "unknown argument".
        if member_profile.runtime.engine != profile.runtime.engine:
            issues.append(Issue(
                "error",
                f"Member '{member_profile.name}' uses the "
                f"{member_profile.runtime.engine} engine but the router runs "
                f"{profile.runtime.engine}; the router spawns members with its own "
                f"engine, which would reject the member's engine-specific flags. "
                f"Set the member's Engine to {profile.runtime.engine}."))

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

        # llama.cpp spawns each member instance on its OWN random loopback port
        # and strips/overrides --port from the preset (verified live b10298:
        # "spawning server instance ... on port 45001"). A port set on a member
        # profile therefore does nothing here -- warn, or a user watching the
        # logs concludes the router itself is randomizing its port. Clients
        # always connect through the router's port.
        if member_profile.settings.get("port") not in (None, DEFAULT_PORT):
            issues.append(Issue(
                "warning",
                f"Member '{member_profile.name}' sets port "
                f"{member_profile.settings['port']}, but the router assigns member "
                f"instances their own internal ports; clients connect through the "
                f"router's port ({profile_port(profile)})."))

    # NB: the non-loopback-without-a-key error is raised in validate() for both
    # modes, so it is deliberately not repeated here.

    models_max = profile.settings.get("models-max", CATALOG["models-max"].default)
    if isinstance(models_max, int) and models_max > 1:
        issues.append(Issue(
            "warning",
            f"models-max is {models_max}: the router may hold that many models resident "
            f"at once, which can exceed VRAM. Use 1 unless the members are small."))


    # Absent OR blank both mean "--cors-origins is omitted from the argv", so
    # the server runs with upstream's default "*": read both as the CATALOG
    # default, else the most dangerous configuration becomes unwarnable
    # (profiles saved by older UI versions stored cors-origins as "").
    origins = str(profile.settings.get("cors-origins")
                  or CATALOG["cors-origins"].default)
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
