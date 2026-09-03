from llama_launcher.core.lora_state import LoraAdapter, parse_adapters


def test_parses_a_normal_payload():
    got = parse_adapters(
        [
            {"id": 0, "path": "/Models/adapters/style.gguf", "scale": 0.0},
            {"id": 1, "path": "/Models/adapters/domain.gguf", "scale": 0.65},
        ]
    )
    assert got == [
        LoraAdapter(id=0, path="/Models/adapters/style.gguf", scale=0.0),
        LoraAdapter(id=1, path="/Models/adapters/domain.gguf", scale=0.65),
    ]


def test_payload_is_a_bare_array_not_a_data_object():
    # /lora-adapters differs from the router's /v1/models here; feeding it the
    # router shape must yield nothing rather than a bogus adapter.
    assert parse_adapters({"data": [{"id": 0}]}) == []


def test_adapter_without_an_integer_id_is_dropped():
    # ids are what a POST addresses, so a defaulted id would rescale the wrong
    # adapter.
    assert parse_adapters([{"path": "x.gguf", "scale": 1.0}]) == []
    assert parse_adapters([{"id": "0", "path": "x.gguf"}]) == []


def test_bool_is_not_accepted_as_an_id_or_scale():
    # bool is an int subclass in Python; True must not become adapter id 1.
    assert parse_adapters([{"id": True, "path": "x.gguf"}]) == []
    got = parse_adapters([{"id": 2, "path": "x.gguf", "scale": True}])
    assert got == [LoraAdapter(id=2, path="x.gguf", scale=0.0)]


def test_missing_scale_defaults_to_zero_meaning_inactive():
    got = parse_adapters([{"id": 3, "path": "x.gguf"}])
    assert got[0].scale == 0.0
    assert got[0].active is False


def test_active_distinguishes_loaded_from_contributing():
    # The resting state under --lora-init-without-apply is loaded-but-zeroed, so
    # "is it loaded" and "is it doing anything" are different questions.
    assert LoraAdapter(id=0, path="a.gguf", scale=0.0).active is False
    assert LoraAdapter(id=0, path="a.gguf", scale=0.01).active is True


def test_name_is_the_filename_and_survives_an_empty_path():
    assert LoraAdapter(id=0, path="/Models/adapters/style.gguf").name == "style.gguf"
    assert LoraAdapter(id=7, path="").name == "7"


def test_junk_entries_are_skipped_not_fatal():
    got = parse_adapters([None, 5, "x", {"id": 1, "path": "ok.gguf", "scale": 1.0}])
    assert [a.id for a in got] == [1]


def test_non_list_payloads_are_empty():
    assert parse_adapters(None) == []
    assert parse_adapters("[]") == []
