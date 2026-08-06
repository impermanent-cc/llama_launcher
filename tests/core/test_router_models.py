from llama_launcher.core.router_models import RouterModel, parse_models


def test_parses_loaded_model():
    payload = {"data": [{
        "id": "qwen", "path": "/models/qwen.gguf",
        "status": {"value": "loaded", "args": ["llama-server", "-ctx", "4096"]},
    }]}
    [m] = parse_models(payload)
    assert m == RouterModel(id="qwen", path="/models/qwen.gguf", status="loaded",
                            progress=None, args=("llama-server", "-ctx", "4096"),
                            failed=False, exit_code=None)


def test_parses_unloaded_model_without_args():
    payload = {"data": [{"id": "a", "status": {"value": "unloaded"}}]}
    [m] = parse_models(payload)
    assert m.status == "unloaded"
    assert m.args == ()
    assert m.path == ""


def test_parses_sleeping_model():
    payload = {"data": [{"id": "a", "status": {"value": "sleeping"}}]}
    assert parse_models(payload)[0].status == "sleeping"


def test_parses_failed_model_with_exit_code():
    payload = {"data": [{"id": "a", "status": {
        "value": "unloaded", "failed": True, "exit_code": 1}}]}
    [m] = parse_models(payload)
    assert m.failed is True
    assert m.exit_code == 1


def test_download_progress_is_a_fraction():
    payload = {"data": [{"id": "a", "status": {
        "value": "downloading",
        "progress": {"https://x/model.gguf": {"done": 50, "total": 200}},
    }}]}
    assert parse_models(payload)[0].progress == 0.25


def test_download_progress_sums_parallel_files():
    payload = {"data": [{"id": "a", "status": {
        "value": "downloading",
        "progress": {"u1": {"done": 1, "total": 4}, "u2": {"done": 1, "total": 4}},
    }}]}
    assert parse_models(payload)[0].progress == 0.25


def test_load_progress_uses_value_field():
    payload = {"data": [{"id": "a", "status": {
        "value": "loading",
        "progress": {"stages": ["text_model"], "current": "text_model", "value": 0.5},
    }}]}
    assert parse_models(payload)[0].progress == 0.5


def test_zero_total_download_does_not_divide_by_zero():
    payload = {"data": [{"id": "a", "status": {
        "value": "downloading", "progress": {"u": {"done": 0, "total": 0}}}}]}
    assert parse_models(payload)[0].progress is None


def test_missing_or_malformed_payload_yields_empty_list():
    assert parse_models({}) == []
    assert parse_models({"data": "nonsense"}) == []
    assert parse_models({"data": [{"no_id": 1}]}) == []
