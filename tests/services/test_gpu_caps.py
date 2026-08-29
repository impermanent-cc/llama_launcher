from llama_launcher.services.gpu import parse_compute_caps


def test_parse_compute_caps():
    assert parse_compute_caps("12.0\n8.6\n") == ["120", "86"]


def test_parse_compute_caps_skips_garbage():
    assert parse_compute_caps("N/A\n\n9.0\n") == ["90"]
