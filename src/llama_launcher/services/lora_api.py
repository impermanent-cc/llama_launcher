"""HTTP client for llama-server's /lora-adapters control plane.

Adapters are chosen at launch (`--lora` / `--lora-scaled`), but their SCALES are
live state: a running server will restretch them without a reload, which is the
only way to A/B an adapter against the base model on one loaded copy of the
weights. Pair it with the `lora-init-without-apply` setting to start with every
adapter loaded but inactive.

Two rules worth stating, because both are easy to get wrong:

1. A POST always carries the FULL adapter list, every id with an explicit scale.
   Upstream documents disabling as "either remove it from the list, or set scale
   to 0", so a partial list is ambiguous by construction; sending all of them
   means the request says exactly what it means regardless of how the server
   resolves omissions.
2. Scales are floats and 0.0 is meaningful (adapter loaded, contributing
   nothing), so nothing here may treat 0.0 as "unset" and skip it.
"""

import requests

from llama_launcher.core.lora_state import LoraAdapter, parse_adapters


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def auth_headers(api_key: str | None) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def list_adapters(host: str, port: int, api_key: str | None,
                  timeout: float = 1.0) -> list[LoraAdapter] | None:
    """Adapters the server loaded, or None when it could not be reached.

    None and [] are different and callers must keep them apart: [] means the
    server is up and was launched with no adapters (so the LoRA controls should
    say so), while None means no answer and the previous view should stand.

    The timeout matches the other UI-thread pollers.
    """
    try:
        r = requests.get(f"{base_url(host, port)}/lora-adapters",
                         headers=auth_headers(api_key), timeout=timeout)
        if r.status_code != 200:
            return None
        return parse_adapters(r.json())
    except (requests.RequestException, ValueError):
        return None


def set_scales(host: str, port: int, api_key: str | None,
               scales: dict[int, float], timeout: float = 10.0) -> bool:
    """Apply `scales` ({adapter id: scale}) to the running server.

    The caller passes every adapter it knows about; see rule 1 in the module
    docstring. Returns False rather than raising so a failed apply degrades to
    "the UI reverts to what the server actually reports" instead of a traceback
    on the UI thread.
    """
    body = [{"id": int(i), "scale": float(s)} for i, s in sorted(scales.items())]
    try:
        r = requests.post(f"{base_url(host, port)}/lora-adapters",
                          headers=auth_headers(api_key), json=body, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False
