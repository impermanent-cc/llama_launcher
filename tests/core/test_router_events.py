from llama_launcher.core.router_events import RouterEvent, parse_sse_event


def test_parses_model_status_event():
    block = 'data: {"model": "qwen", "event": "model_status", "data": {"status": "loading"}}'
    ev = parse_sse_event(block)
    assert ev == RouterEvent(model="qwen", event="model_status",
                             data={"status": "loading"})


def test_parses_multiline_data_block():
    block = 'data: {"model": "a", "event": "model_remove",\ndata:  "data": {}}'
    ev = parse_sse_event(block)
    assert ev.event == "model_remove"


def test_ignores_comment_pings():
    assert parse_sse_event(": ping") is None


def test_ignores_empty_block():
    assert parse_sse_event("") is None
    assert parse_sse_event("\n\n") is None


def test_returns_none_on_malformed_json():
    assert parse_sse_event("data: {not json") is None


def test_parses_models_reload_broadcast():
    block = 'data: {"model": "*", "event": "models_reload"}'
    ev = parse_sse_event(block)
    assert ev.model == "*"
    assert ev.data == {}
