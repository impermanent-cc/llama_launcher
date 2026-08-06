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


def parse_labeled_metric(text: str, name: str, label: str) -> dict[str, float]:
    """Return {label_value: sample} for one labeled Prometheus series.

    `parse_metrics` strips labels, so labeled families like
    llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}
    collapse onto a single key there. This keeps them apart.
    """
    out: dict[str, float] = {}
    needle = f'{label}="'
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith(name + "{"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        labels = parts[0][len(name) + 1:parts[0].rfind("}")] if "}" in parts[0] else ""
        start = labels.find(needle)
        if start < 0:
            continue
        start += len(needle)
        end = labels.find('"', start)
        if end < 0:
            continue
        try:
            out[labels[start:end]] = float(parts[1])
        except ValueError:
            continue
    return out
