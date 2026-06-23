from llama_launcher.core.prometheus import parse_metrics

SAMPLE = """# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 42.5
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
llamacpp:prompt_tokens_total{model="x"} 1234
garbage line
llamacpp:bad notanumber
"""


def test_parse_metrics():
    m = parse_metrics(SAMPLE)
    assert m["llamacpp:predicted_tokens_seconds"] == 42.5
    assert m["llamacpp:requests_processing"] == 1.0
    assert m["llamacpp:prompt_tokens_total"] == 1234.0   # labels stripped
    assert "llamacpp:bad" not in m                        # non-numeric skipped
