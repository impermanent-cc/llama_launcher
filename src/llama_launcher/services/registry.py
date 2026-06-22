import re

import requests

_BUILD_RE = re.compile(r"^(?P<prefix>.+)-b(?P<num>\d+)$")
_TOKEN_URL = "https://ghcr.io/token?scope=repository:{repo}:pull"
_TAGS_URL = "https://ghcr.io/v2/{repo}/tags/list?n=1000"


def split_image(image: str) -> tuple[str, str]:
    repo, _, tag = image.rpartition(":")
    if "/" in tag or not repo:  # no tag present
        return image, ""
    return repo, tag


def variant_prefix(tag: str) -> str:
    m = _BUILD_RE.match(tag)
    return m.group("prefix") if m else tag


def latest_build_tag(tags: list[str], prefix: str) -> str | None:
    best_num = -1
    best_tag = None
    for t in tags:
        m = _BUILD_RE.match(t)
        if not m or m.group("prefix") != prefix:
            continue
        num = int(m.group("num"))
        if num > best_num:
            best_num, best_tag = num, t
    return best_tag


def _ghcr_repo(repo: str) -> str:
    # Strip a leading "ghcr.io/" host if present for the API path.
    return repo.split("ghcr.io/", 1)[-1]


def fetch_latest(repo: str, prefix: str, timeout: float = 10.0) -> str | None:
    api_repo = _ghcr_repo(repo)
    token = requests.get(_TOKEN_URL.format(repo=api_repo), timeout=timeout).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    url = _TAGS_URL.format(repo=api_repo)
    all_tags: list[str] = []
    while url:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        all_tags += resp.json().get("tags") or []
        link = resp.headers.get("Link", "")
        m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        url = ("https://ghcr.io" + m.group(1)) if m else None
    return latest_build_tag(all_tags, prefix)
