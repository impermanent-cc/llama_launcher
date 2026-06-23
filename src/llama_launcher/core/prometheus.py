def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        if "{" in name:
            name = name[:name.index("{")]
        try:
            out[name] = float(parts[1])
        except ValueError:
            continue
    return out
