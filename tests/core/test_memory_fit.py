import json
from dataclasses import replace

from llama_launcher.core import memory_fit as mf
from llama_launcher.core.gguf import GgufMeta, TensorInfo

MIB = 1024 * 1024
GIB = 1024 * MIB


def _meta(n_layers=8, moe=False, swa=None):
    ts = [TensorInfo("token_embd.weight", 1, 0, 10 * MIB)]
    for i in range(n_layers):
        ts.append(TensorInfo(f"blk.{i}.attn_q.weight", 1, 0, 100 * MIB))
        name = f"blk.{i}.ffn_up_exps.weight" if moe else f"blk.{i}.ffn_up.weight"
        ts.append(TensorInfo(name, 1, 0, 400 * MIB))
    ts.append(TensorInfo("output.weight", 1, 0, 100 * MIB))
    return GgufMeta(
        arch="qwen3moe" if moe else "llama",
        n_layers=n_layers,
        n_head=8,
        n_head_kv=8,
        n_embd=512,
        ctx_train=32768,
        n_ff=256,
        n_vocab=1000,
        expert_count=64 if moe else None,
        sliding_window=swa,
        tensors=tuple(ts),
    )


def _report(
    meta=None,
    free=(16 * GIB,),
    ram=64 * GIB,
    engine="llama.cpp",
    with_details=True,
    **settings,
):
    settings = {"ctx-size": 4096, **settings}
    if settings.get("ctx-size") is None:
        settings.pop("ctx-size", None)
    return mf.fit_report(
        meta or _meta(),
        0,
        settings=settings,
        engine=engine,
        free_bytes_per_gpu=list(free),
        ram_available=ram,
        with_details=with_details,
    )


def test_fits_has_no_messages():
    r = _report()
    assert r.fits and r.messages == ()
    assert r.cards[0].fits and r.cards[0].margin > 0
    assert r.ram.fits is True


def test_smallest_fitting_offload_none_when_zero_already_fits():
    meta = _meta()
    found = mf.smallest_fitting_offload(
        meta,
        0,
        settings={"ctx-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB],
    )
    assert found is None


def test_fit_report_none_without_cards():
    r = mf.fit_report(
        _meta(),
        0,
        settings={"ctx-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[],
        ram_available=64 * GIB,
    )
    assert r is None


def test_shortfall_names_card_and_smallest_n_cpu_ffn():
    # 8 layers of 500 MiB plus output and overhead: about 4.6 GiB; give 3 GiB
    r = _report(free=(3 * GIB,), fit="off")
    assert not r.fits
    text = r.messages[0].text
    assert "GPU0" in text and "exceeds" in text
    assert "--n-cpu-ffn 4" in text or "--n-cpu-ffn 5" in text
    assert r.messages[0].dialog is True


def test_shortfall_moe_suggests_n_cpu_moe():
    r = _report(_meta(moe=True), free=(3 * GIB,), fit="off")
    assert "--n-cpu-moe" in r.messages[0].text


def test_moe_detected_from_tensor_names_without_expert_count():
    meta = replace(_meta(moe=True), expert_count=None)
    r = _report(meta, free=(3 * GIB,), fit="off")
    assert "--n-cpu-moe" in r.messages[0].text


def test_shortfall_on_ik_dense_gives_override_tensor_alternation():
    r = _report(free=(3 * GIB,), engine="ik_llama.cpp", **{"n-gpu-layers": 99})
    text = r.messages[0].text
    assert "--override-tensor" in text
    assert r"blk\.(0|1|2|3" in text and r"\.ffn_(up|down|gate)\.=CPU" in text


def test_smallest_fitting_offload_override_tensor_appends_to_existing():
    """The override-tensor candidate the search builds keeps the user's own
    entries first, comma-separated, so the search baseline (n=0) still
    applies them and the suggested value never drops what the user set."""
    meta = _meta()
    existing = r"blk\.0\.ffn_up\.=CPU"
    found = mf.smallest_fitting_offload(
        meta,
        0,
        settings={"n-gpu-layers": 99, "override-tensor": existing},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[3 * GIB],
    )
    assert found is not None
    key, value = found
    assert key == "override-tensor"
    assert value.startswith(existing + ",")


def test_smallest_fitting_offload_override_tensor_strips_trailing_comma():
    """A trailing comma (with or without trailing whitespace) on the user's
    existing override-tensor value must not leave an empty entry in the
    joined candidate."""
    meta = _meta()
    existing = r"blk\.0\.ffn_up\.=CPU"
    found = mf.smallest_fitting_offload(
        meta,
        0,
        settings={"n-gpu-layers": 99, "override-tensor": existing + ", "},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[3 * GIB],
    )
    assert found is not None
    _key, value = found
    assert value.startswith(existing + ",")
    assert ",," not in value


def test_no_count_fits_says_so():
    r = _report(free=(1 * GIB,), fit="off")
    text = r.messages[0].text
    assert "Even offloading every layer leaves the card over budget" in text
    assert "no offload count" in text


def test_fit_unset_predicts_context_and_reaches_dialog():
    """With no --ctx-size, the model's trained context supplies the context
    the shrink prediction is solved from."""
    r = _report(free=(4 * GIB,), **{"ctx-size": None})
    text = r.messages[0].text
    assert "shrink" in text and "context" in text
    assert r.messages[0].dialog is True


def test_fit_on_explicit_stays_out_of_dialog():
    r = _report(free=(4 * GIB,), fit="on", **{"ctx-size": None})
    assert "shrink" in r.messages[0].text
    assert r.messages[0].dialog is False


def test_fit_is_inert_when_ngl_or_override_is_set():
    for extra in ({"n-gpu-layers": 99}, {"override-tensor": "exps=CPU"}):
        r = _report(free=(4 * GIB,), **{"ctx-size": None, **extra})
        assert "shrink" not in r.messages[0].text
        assert r.messages[0].dialog is True


def test_fit_is_inert_when_a_cpu_offload_flag_is_set():
    for extra in ({"n-cpu-ffn": 4}, {"cpu-moe": True}):
        r = _report(free=(3 * GIB,), **{"ctx-size": None, **extra})
        assert "shrink" not in r.messages[0].text
        assert r.messages[0].dialog is True


def test_fit_is_inert_when_tensor_split_is_set():
    r = _report(free=(2 * GIB, 2 * GIB), **{"ctx-size": None, "tensor-split": "60,40"})
    assert "shrink" not in r.messages[0].text
    assert r.messages[0].dialog is True


def test_fit_is_inert_when_split_mode_is_not_layer_with_two_cards():
    r = _report(free=(2 * GIB, 2 * GIB), **{"ctx-size": None, "split-mode": "row"})
    assert "shrink" not in r.messages[0].text
    assert r.messages[0].dialog is True


def test_fit_json_false_maps_to_off():
    r = _report(free=(4 * GIB,), fit=False, **{"ctx-size": None})
    assert "shrink" not in r.messages[0].text
    assert r.messages[0].dialog is True


def test_fit_active_with_ctx_size_set_keeps_context_and_moves_layers():
    """A profile that sets --ctx-size keeps that context under an active
    fit: llama.cpp moves whole layers to RAM instead of shrinking it, and
    the note names no predicted context."""
    r = _report(free=(4 * GIB,))  # default ctx-size 4096 from _report
    text = r.messages[0].text
    assert "shrink" not in text
    assert "keep" in text and "context" in text and "RAM" in text
    assert r.messages[0].dialog is True


def test_fit_active_with_ctx_size_set_names_experts_first_on_moe():
    r = _report(_meta(moe=True), free=(4 * GIB,))
    assert "expert" in r.messages[0].text


def test_predicted_fit_ctx_none_when_ctx_size_set():
    meta = _meta()
    assert (
        mf.predicted_fit_ctx(
            meta,
            0,
            settings={"ctx-size": 4096},
            engine="llama.cpp",
            free_bytes_per_gpu=[1 * GIB],
        )
        is None
    )


def test_predicted_fit_ctx_floor_then_layers():
    meta = _meta()  # no --ctx-size: the trained context (32768) is the top
    # 6 GiB card, 1024 MiB margin: solve between 4096 and the trained context
    ctx, drops = mf.predicted_fit_ctx(
        meta,
        0,
        settings={},
        engine="llama.cpp",
        free_bytes_per_gpu=[6 * GIB],
        raw_args="",
    )
    assert 4096 <= ctx < 32768 and ctx % 256 == 0
    # 2 GiB card, 1024 MiB margin: even the floor context does not fit
    ctx, drops = mf.predicted_fit_ctx(
        meta,
        0,
        settings={},
        engine="llama.cpp",
        free_bytes_per_gpu=[2 * GIB],
        raw_args="",
    )
    assert ctx == 4096 and drops is True


def test_ik_fit_on_moe_note_and_dense_warning():
    moe = _report(
        _meta(moe=True),
        free=(3 * GIB,),
        engine="ik_llama.cpp",
        fit="on",
        **{"n-gpu-layers": 99},
    )
    assert "experts" in moe.messages[0].text and moe.messages[0].dialog is False
    dense = _report(
        free=(3 * GIB,), engine="ik_llama.cpp", fit="on", **{"n-gpu-layers": 99}
    )
    assert "refuses" in dense.messages[0].text and dense.messages[0].dialog is True


def test_ik_fit_on_moe_grows_ram_and_fits():
    meta = _meta(moe=True)
    off = _report(
        meta, free=(3 * GIB,), engine="ik_llama.cpp", fit="off", **{"n-gpu-layers": 99}
    )
    on = _report(
        meta, free=(3 * GIB,), engine="ik_llama.cpp", fit="on", **{"n-gpu-layers": 99}
    )
    assert on.fits is True
    assert on.ram.est > off.ram.est
    text = on.messages[0].text
    assert "experts" in text and "layers" in text
    found = mf.smallest_fitting_offload(
        meta,
        0,
        settings={"ctx-size": 4096, "fit": "on", "n-gpu-layers": 99},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[3 * GIB],
    )
    _, layers = found
    assert f"{layers} layer" in text


def test_ik_fit_on_moe_still_over_when_even_all_experts_in_ram_do_not_fit():
    r = _report(
        _meta(moe=True),
        free=(1 * GIB,),
        engine="ik_llama.cpp",
        fit="on",
        **{"n-gpu-layers": 99},
    )
    assert r.fits is False


def test_ik_fit_on_moe_without_tensor_table_falls_through_to_plain_shortfall():
    """A MoE model on the file-size fallback (no tensor table) has no
    expert bytes to move, so ik's --fit-on note does not apply and the
    plain shortfall wording is used instead."""
    meta = replace(_meta(moe=True), tensors=())
    r = _report(
        meta,
        free=(100 * MIB,),
        engine="ik_llama.cpp",
        fit="on",
        **{"n-gpu-layers": 99},
    )
    assert r.fits is False
    text = r.messages[0].text
    assert "experts" not in text and "refuses" not in text
    assert "may not fit" in text


def test_ik_fit_off_gives_plain_shortfall():
    r = _report(
        free=(3 * GIB,), engine="ik_llama.cpp", fit="off", **{"n-gpu-layers": 99}
    )
    text = r.messages[0].text
    assert "experts" not in text and "refuses" not in text


def test_ram_warning_wording_by_load_mode():
    plain = _report(ram=100 * MIB, **{"n-cpu-ffn": 8})
    assert plain.ram.fits is False
    assert any("page" in m.text for m in plain.messages)
    locked = _report(ram=100 * MIB, **{"n-cpu-ffn": 8, "load-mode": "mlock"})
    assert any("fail" in m.text for m in locked.messages)
    assert all(m.dialog for m in locked.messages)


def test_ram_wording_ignores_load_mode_on_engine_that_lacks_it():
    # ik_llama.cpp does not accept --load-mode; "none" must not read as locked
    r = _report(ram=100 * MIB, engine="ik_llama.cpp", **{"load-mode": "none"})
    assert any("page" in m.text for m in r.messages)


def test_ram_unknown_is_not_a_failure():
    r = _report(ram=None)
    assert r.ram.fits is None and r.fits


def test_render_lines_tooltip_dialog_json():
    r = _report(_meta(swa=1024), free=(16 * GIB, 8 * GIB), **{"tensor-split": "60,40"})
    lines = mf.render_lines(r)
    assert lines[0].startswith("GPU0") and lines[1].startswith("GPU1")
    assert lines[2].startswith("RAM")
    assert "up to" in lines[0] or "up to" in lines[1]
    assert "~" in lines[0]
    tip = mf.render_tooltip(r)
    assert "formula" in tip and "upper bound" in tip
    assert mf.render_dialog(r) is None
    j = mf.to_json(r)
    assert (
        j["fits"] is True and len(j["cards"]) == 2 and j["ram"]["available"] == 64 * GIB
    )
    over = _report(free=(3 * GIB,), fit="off")
    dialog = mf.render_dialog(over)
    assert dialog.startswith("GPU0")
    assert "RAM:" in dialog and over.messages[0].text in dialog
    assert "<" not in dialog and "&" not in dialog
    assert "color" in mf.render_lines(over)[0]


def test_plain_lines_have_no_markup():
    over = _report(free=(3 * GIB,), fit="off")
    lines = mf.plain_lines(over)
    text = "\n".join(lines)
    assert "<" not in text and "&" not in text
    assert all(line.isascii() for line in lines)


def test_to_json_is_json_serializable():
    r = _report()
    assert json.loads(json.dumps(mf.to_json(r)))["fits"] is True


def test_fit_report_none_without_layers():
    meta = replace(_meta(), n_layers=None)
    r = mf.fit_report(
        meta,
        0,
        settings={"ctx-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB],
        ram_available=64 * GIB,
    )
    assert r is None


def _hybrid_meta(n_layers=40):
    """A hybrid model: a full-attention layer every fourth, recurrent state
    sizes on the rest."""
    return replace(
        _meta(n_layers=n_layers),
        full_attention_interval=4,
        head_dim_k=256,
        head_dim_v=256,
        ssm_conv_kernel=4,
        ssm_inner_size=4096,
        ssm_state_size=128,
        ssm_group_count=16,
    )


def test_json_and_lines_carry_state_and_checkpoints():
    """A hybrid model's recurrent state reaches the JSON card dict and the
    card readout lines; the checkpoints are a RAM key and a RAM line, since
    the server keeps them in host memory."""
    report = _report(_hybrid_meta(), free=(16 * GIB, 8 * GIB))
    d = mf.to_json(report)
    assert "state" in d["cards"][0] and "checkpoints" not in d["cards"][0]
    assert {"state", "checkpoints"} <= set(d["ram"])
    assert d["cards"][0]["state"] > 0
    card_state = sum(c["state"] for c in d["cards"])
    assert d["ram"]["checkpoints"] == 32 * (card_state + d["ram"]["state"])
    lines = mf.plain_lines(report)
    assert "state" in lines[0] and "checkpoints" not in lines[0]
    assert "state" in lines[-1] and "checkpoints" in lines[-1]


def test_report_has_no_balanced_split_when_not_asked():
    """A fitting readout leaves FitReport.balanced None and carries no
    balanced_split key, so it pays nothing for a search it will not show."""
    r = _report(free=(16 * GIB, 16 * GIB))
    assert r.balanced is None and r.messages == ()
    assert "balanced_split" not in json.loads(json.dumps(mf.to_json(r)))


def test_asking_for_the_split_does_not_move_the_estimate():
    """The balanced split is a suggestion: what the estimate assumes about
    an unset --tensor-split is the engines' free-VRAM proportion either
    way."""
    kw = dict(
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 8 * GIB],
        ram_available=64 * GIB,
    )
    plain = mf.fit_report(_meta(n_layers=16), 0, **kw)
    asked = mf.fit_report(_meta(n_layers=16), 0, with_balanced=True, **kw)
    assert [c.est for c in plain.cards] == [c.est for c in asked.cards]
    assert plain.messages == asked.messages


def test_with_balanced_carries_the_split_into_the_json():
    """--estimate --json carries the balanced split as balanced_split with
    its value, layers_per_card, boundary_layers and fits keys; fits is the
    split's own outcome, not the report's, so a suggestion that resolves a
    shortfall reports True while the plain readout still reports False."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
        with_balanced=True,
    )
    obj = mf.to_json(r)["balanced_split"]
    assert obj["value"] == ",".join(str(n) for n in obj["layers_per_card"])
    assert sum(obj["layers_per_card"]) == 17
    assert len(obj["boundary_layers"]) == 1
    assert isinstance(obj["layers_per_card"], list)
    assert isinstance(obj["boundary_layers"], list)
    assert r.fits is False
    assert obj["fits"] is True


def test_no_balanced_split_key_in_row_mode():
    """balanced_split is absent whenever balance.balanced_split computes no
    split, which split-mode row falls under."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "split-mode": "row"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 8 * GIB],
        ram_available=64 * GIB,
        with_balanced=True,
    )
    assert r.balanced is None and "balanced_split" not in mf.to_json(r)


def test_a_shortfall_computes_the_split_without_being_asked():
    """fit_report computes the balanced split whenever a card is over
    budget, even when with_balanced was never passed."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    assert any(not c.fits for c in r.cards)
    assert r.balanced is not None


def test_three_cards_get_every_boundary_and_a_three_way_split():
    """A balanced split across three cards carries three layer counts and
    two boundaries, in both the dataclass and the JSON it feeds, and its
    layer counts sum to the entries actually placed on cards."""
    from llama_launcher.core import balance

    settings = {"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "10,1,1"}
    meta = _meta(n_layers=12)
    r = mf.fit_report(
        meta,
        0,
        settings=settings,
        engine="llama.cpp",
        free_bytes_per_gpu=[3 * GIB, 3 * GIB, 3 * GIB],
        ram_available=64 * GIB,
    )
    assert any(not c.fits for c in r.cards)
    assert r.balanced is not None
    entries = balance.gpu_entries(meta, settings=settings, engine="llama.cpp")
    assert sum(r.balanced.layers_per_card) == len(entries)
    assert len(r.balanced.layers_per_card) == 3
    assert len(r.balanced.boundary_layers) == 2
    js = mf.to_json(r)["balanced_split"]
    assert len(js["layers_per_card"]) == 3 and len(js["boundary_layers"]) == 2


def test_draft_and_projector_bytes_reach_the_balanced_search(monkeypatch):
    """fit_report forwards draft_meta, draft_weights and mmproj_bytes to the
    balanced search."""
    from llama_launcher.core import balance

    seen = {}
    real = balance.balanced_split

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(mf._balance, "balanced_split", spy)
    draft = _meta(n_layers=4)
    mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
        draft_meta=draft,
        draft_weights=123,
        mmproj_bytes=456,
    )
    assert seen["draft_meta"] is draft
    assert seen["draft_weights"] == 123
    assert seen["mmproj_bytes"] == 456


def test_shortfall_names_the_balanced_split_that_fits_on_its_own():
    """A card shortfall message names the balanced split, and its own
    fitting is stated, when the split resolves the shortfall by itself."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    assert r.fits is False
    assert r.balanced is not None and r.balanced.fits is True
    text = " ".join(m.text for m in r.messages)
    assert "--tensor-split " + r.balanced.value in text
    assert "fits every card" in text
    assert "replaces" not in text
    assert f"card boundary at layer {r.balanced.boundary_layers[0]}" in text


def test_offload_count_stays_at_the_profile_split_when_balanced_fits_alone():
    """When the balanced split differs from the profile's own resolved
    placement and fits every card on its own, the message names the split
    and carries no offload count clause at all."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    r = mf.fit_report(meta, 0, **kw)
    assert r.balanced is not None and r.balanced.fits is True
    text = " ".join(m.text for m in r.messages)
    assert "Balanced --tensor-split" in text
    assert "no offload count fits" not in text
    assert "Smallest offload" not in text


def test_offload_count_is_searched_at_the_balanced_split():
    """When the balanced split does not fit on its own and needs a smaller
    offload count than the profile's own split does, the message names
    the count found at the balanced split rather than at the profile's."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "tensor-split": "15,2",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[3 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    r = mf.fit_report(meta, 0, **kw)
    assert r.balanced is not None and r.balanced.fits is False
    at_balanced = mf.smallest_fitting_offload(
        meta,
        0,
        settings={**kw["settings"], "tensor-split": r.balanced.value},
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    at_profile = mf.smallest_fitting_offload(
        meta,
        0,
        settings=kw["settings"],
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    assert at_balanced is not None and at_profile is not None
    assert at_balanced[1] < at_profile[1]
    key, value = at_balanced
    text = " ".join(m.text for m in r.messages)
    assert f"--{key} {value}" in text


def test_offload_count_search_ignores_a_raw_tensor_split():
    """A --tensor-split carried in raw args, which would otherwise win
    back over the balanced overlay once effective settings are recomputed,
    does not pull the offload count search back to the profile's split."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[3 * GIB, 3 * GIB],
        ram_available=64 * GIB,
    )
    r = mf.fit_report(meta, 0, raw_args="--tensor-split 90,10", **kw)
    assert r.balanced is not None and r.balanced.fits is False
    at_balanced = mf.smallest_fitting_offload(
        meta,
        0,
        settings={**kw["settings"], "tensor-split": r.balanced.value},
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    at_raw_split = mf.smallest_fitting_offload(
        meta,
        0,
        settings=kw["settings"],
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
        raw_args="--tensor-split 90,10",
    )
    assert at_balanced is not None and at_balanced != at_raw_split
    key, value = at_balanced
    text = " ".join(m.text for m in r.messages)
    assert f"--{key} {value}" in text
    assert f"--{at_raw_split[0]} {at_raw_split[1]}" not in text
    assert "--tensor-split " + r.balanced.value in text


def test_offload_count_names_the_split_it_was_found_at():
    """When the smallest fitting count found at the balanced split is
    smaller than the count the profile's own split needs, the message
    names the split alongside the count, since the count is minimal only
    there."""
    meta = _meta(n_layers=16)
    free = [4 * GIB, 4 * GIB]
    settings = {"tensor-split": "15,2"}
    r = mf.fit_report(
        meta,
        0,
        settings=settings,
        engine="llama.cpp",
        free_bytes_per_gpu=free,
        ram_available=64 * GIB,
    )
    assert r.balanced is not None and r.balanced.fits is False
    at_profile = mf.smallest_fitting_offload(
        meta, 0, settings=settings, engine="llama.cpp", free_bytes_per_gpu=free
    )
    key, value = at_profile
    text = " ".join(m.text for m in r.messages)
    assert f"--{key} {value}" not in text
    assert f"at --tensor-split {r.balanced.value}" in text


def test_shortfall_names_the_split_it_replaces():
    """A profile that already sets --tensor-split gets a message stating
    the balanced suggestion replaces that value."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "tensor-split": "60,40",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert "60,40" in text and "replaces" in text
    assert "--fit" not in text


def test_balanced_sentence_follows_the_shortfall_wording_not_precedes_it():
    """The balanced-split sentence comes after the branch's own shortfall
    wording, so a split that fits every card never reads as if it were the
    subject of a following "may not fit"."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "tensor-split": "60,40",
        },
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert text.index("may not fit") < text.index("Balanced --tensor-split")


def test_shortfall_warns_that_a_split_turns_fit_off_on_mainline():
    """On mainline, with nothing already claiming the offload or the split
    fit would otherwise make, the shortfall message states that setting the
    suggested --tensor-split keeps --fit from acting."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    assert "keeps --fit from acting" in " ".join(m.text for m in r.messages)


def test_shortfall_omits_the_fit_note_on_ik():
    """With --fit explicitly on and no CPU offload flag, override or
    tensor-split of its own claiming the placement, the balanced-split
    sentence on ik_llama.cpp still carries no clause about --fit
    deactivating: that note belongs to mainline's fit search alone."""
    r = mf.fit_report(
        _meta(n_layers=16, moe=False),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "fit": "on"},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert "--tensor-split " + r.balanced.value in text
    assert "keeps --fit from acting" not in text


def test_balanced_split_text_stays_out_of_the_ram_message():
    """The balanced-split sentence belongs to the card shortfall message
    alone: a report whose cards fit but whose RAM does not carries a
    balanced split without adding its sentence to the RAM message."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[24 * GIB, 24 * GIB],
        ram_available=50 * MIB,
        with_balanced=True,
    )
    assert all(c.fits for c in r.cards) and r.ram.fits is False
    assert r.balanced is not None
    text = " ".join(m.text for m in r.messages)
    assert "--tensor-split" not in text


def test_offload_count_names_the_profile_split_when_balanced_needs_more():
    """When the balanced split does not fit on its own and needs a larger
    offload count than the profile's own split does, the message names the
    profile's own count rather than the balanced pair, since naming the
    larger count would suggest a split that makes the shortfall worse."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={"ctx-size": 16384, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[2.0 * GIB, 2.0 * GIB],
        ram_available=64 * GIB,
    )
    r = mf.fit_report(meta, 0, **kw)
    assert r.balanced is not None and r.balanced.fits is False
    at_profile = mf.smallest_fitting_offload(
        meta,
        0,
        settings=kw["settings"],
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    at_balanced = mf.smallest_fitting_offload(
        meta,
        0,
        settings={**kw["settings"], "tensor-split": r.balanced.value},
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    assert at_balanced is not None and at_profile is not None
    assert at_balanced[1] > at_profile[1]
    key, value = at_profile
    text = " ".join(m.text for m in r.messages)
    assert f"--{key} {value}" in text
    assert f"at --tensor-split {r.balanced.value}" not in text
    assert "Balanced --tensor-split" not in text


def test_balanced_split_not_named_when_it_matches_the_resolved_placement():
    """A card shortfall message never suggests the balanced split where
    applying it would change nothing: with no --tensor-split set, the
    search's own starting point is the free-VRAM proportion the profile
    already resolves to, and finding no better candidate there must not
    be presented as a suggestion to replace it, nor pair it with the
    warning that an applied split keeps --fit from acting."""
    meta = _meta(n_layers=16)
    free = [1610612736, 2147483648]
    r = mf.fit_report(
        meta,
        0,
        settings={"ctx-size": 8192},
        engine="llama.cpp",
        free_bytes_per_gpu=free,
        ram_available=64 * GIB,
    )
    assert r.balanced is not None and r.balanced.fits is False
    assert r.balanced.layers_per_card == r.balanced.start_layers_per_card
    text = " ".join(m.text for m in r.messages)
    assert "Balanced --tensor-split" not in text
    assert "keeps --fit from acting" not in text


def test_a_shortfall_report_walks_the_tensor_table_once(monkeypatch):
    """One fit_report, balanced search and offload searches included, walks
    the model's tensor table once."""
    from llama_launcher.core import placement as pl

    calls = []
    real = pl._compute_layer_sums

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pl, "_compute_layer_sums", counting)
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "15,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[3 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    assert any(not c.fits for c in r.cards)
    assert r.balanced is not None
    assert len(calls) == 1


def test_a_fitting_balanced_split_carries_no_offload_count_clause():
    """Where the named split fits on its own, the message neither says no
    count fits nor names one."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "15,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert r.balanced.fits
    assert "Balanced --tensor-split" in text
    assert "no offload count fits" not in text
    assert "Smallest offload" not in text


def test_the_balanced_sentence_names_every_boundary():
    """A three-card balanced split names both card boundaries in one
    sentence."""
    r = mf.fit_report(
        _meta(n_layers=12),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "10,1,1"},
        engine="llama.cpp",
        free_bytes_per_gpu=[4 * GIB, 4 * GIB, 4 * GIB],
        ram_available=64 * GIB,
    )
    assert r.balanced.fits
    text = " ".join(m.text for m in r.messages)
    a, b = r.balanced.boundary_layers
    assert f"boundaries at layers {a}, {b}" in text


def test_the_ik_moe_fit_branch_names_the_balanced_split():
    """When ik_llama.cpp still leaves a card short after keeping every
    layer's experts in RAM, and the balanced split differs from the
    profile's own and fits, its sentence joins the shortfall message."""
    r = mf.fit_report(
        _meta(n_layers=16, moe=True),
        0,
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "tensor-split": "15,1",
            "fit": "on",
        },
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[1.6 * GIB, 7.8 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert "experts" in text
    assert "Balanced --tensor-split" in text
    assert r.balanced is not None and r.balanced.fits


def test_the_ik_moe_fit_branch_omits_a_balanced_split_that_does_not_fit():
    """When the balanced split leaves a card short as well, the ik MoE
    shortfall message carries no balanced-split sentence."""
    r = mf.fit_report(
        _meta(n_layers=16, moe=True),
        0,
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "tensor-split": "15,1",
            "fit": "on",
        },
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[0.3 * GIB, 0.3 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert r.balanced is not None and not r.balanced.fits
    assert "Balanced --tensor-split" not in text


def test_override_suggestion_is_quoted_and_whole():
    """The --override-tensor suggestion renders shell-quoted, in single
    quotes for a plain value, as the whole value the search evaluated."""
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={
            "ctx-size": 8192,
            "n-gpu-layers": "all",
            "override-tensor": r"attn_q=CPU",
        },
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[6 * GIB],
        ram_available=64 * GIB,
    )
    text = " ".join(m.text for m in r.messages)
    assert "--override-tensor 'attn_q=CPU,blk\\.(" in text
    assert text.count("'") == 2


def test_override_suggestion_survives_a_quote_in_the_value():
    """An override-tensor value carrying a single quote of its own is
    shell-quoted so shlex.split reads the flag's argument back unchanged."""
    import shlex

    value = "attn_q='CPU',blk.(0)=CPU"
    text = mf._suggestion_text(("override-tensor", value))
    flag = "--override-tensor "
    shown = text[text.index(flag) + len(flag) :].rstrip(".")
    assert shlex.split(shown) == [value]


def test_small_shortfalls_render_in_mib():
    """A shortfall under one gibibyte renders in whole MiB, never rounding
    down to a misleading ~0.0 GiB."""
    assert mf._amount(50 * MIB) == "~50 MiB"
    assert mf._amount(1023 * MIB) == "~1023 MiB"
    assert mf._amount(1023.7 * MIB) == "~1.0 GiB"
    assert mf._amount(GIB) == "~1.0 GiB"
    r = _report(_meta(n_layers=8), free=(4400 * MIB,))
    text = " ".join(m.text for m in r.messages)
    assert "by ~0.0 GiB" not in text
    assert "MiB" in text


def test_uncounted_files_produce_dialog_messages():
    """A draft model or projector under no configured folder counts as zero
    bytes and produces a dialog message naming the setting and the path."""
    r = mf.fit_report(
        _meta(n_layers=8),
        0,
        settings={"ctx-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB],
        ram_available=64 * GIB,
        uncounted=(("Draft model", "/models/d.gguf"), ("Projector", "/models/p.gguf")),
    )
    texts = [m.text for m in r.messages if m.dialog]
    assert (
        "Draft model /models/d.gguf lies under no configured folder; "
        "its bytes are not counted." in texts
    )
    assert (
        "Projector /models/p.gguf lies under no configured folder; "
        "its bytes are not counted." in texts
    )
    assert mf.render_dialog(r) is not None


def test_an_ik_dense_shortfall_report_shares_one_walk_across_the_search(monkeypatch):
    """A dense model on the engine without --n-cpu-ffn, whose offload search
    suggests an --override-tensor alternation instead, costs two tensor-table
    walks for the whole report: one for the profile's own placement and one
    shared by every count the search tries. The generated pattern is one the
    placement's trailing-count reader recognises, so no count pays for a walk
    of its own."""
    from llama_launcher.core import placement as pl

    calls = []
    real = pl._compute_layer_sums

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pl, "_compute_layer_sums", counting)
    r = mf.fit_report(
        _meta(n_layers=16),
        0,
        settings={"ctx-size": 8192, "n-gpu-layers": "all", "tensor-split": "15,1"},
        engine="ik_llama.cpp",
        free_bytes_per_gpu=[3 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    assert any(not c.fits for c in r.cards)
    assert "--override-tensor" in " ".join(m.text for m in r.messages)
    assert len(calls) == 2


def test_kv_per_1k_is_the_marginal_cost_per_device():
    r = _report(free=(16 * GIB, 8 * GIB), **{"tensor-split": "40,60"})
    est = r.estimate
    assert len(r.kv_per_1k) == 3
    per_layer_1k = 2 * 8 * (512 // 8) * 1024 * 2  # K and V, 8 heads, 64 dims, f16
    assert r.kv_per_1k[0] == 4 * per_layer_1k
    assert r.kv_per_1k[1] == 4 * per_layer_1k
    assert r.kv_per_1k[2] == 0
    assert sum(r.kv_per_1k) * 4 == est.cards[0].kv + est.cards[1].kv


def test_kv_per_1k_host_layers_land_on_the_ram_entry():
    r = _report(**{"n-gpu-layers": 5})
    assert r.kv_per_1k[0] > 0 and r.kv_per_1k[1] > 0
    assert r.kv_per_1k[0] == r.kv_per_1k[1]


def test_kv_per_1k_is_zero_past_a_sliding_window():
    # Four default slots share the context, so a slot's share passes the
    # window plus micro-batch (1024 + 512) only from 8192 tokens on.
    meta = replace(
        _meta(swa=1024), sliding_window_pattern=tuple(True for _ in range(8))
    )
    r = _report(meta, **{"ctx-size": 8192})
    assert r.kv_per_1k == (0, 0)
    assert r.estimate.cards[0].kv > 0
    below = _report(meta, **{"ctx-size": 4096})
    assert below.kv_per_1k[0] > 0


def test_kv_per_1k_follows_each_cards_own_layers():
    r = _report(free=(16 * GIB, 8 * GIB), **{"tensor-split": "60,40"})
    per_layer_1k = 2 * 8 * (512 // 8) * 1024 * 2
    assert r.kv_per_1k == (6 * per_layer_1k, 2 * per_layer_1k, 0)


def test_render_lines_open_the_bracket_with_the_layer_range():
    r = _report(free=(16 * GIB, 8 * GIB), **{"tensor-split": "40,60"})
    lines = mf.render_lines(r)
    assert "(layers 0 to 3, weights" in lines[0]
    assert "(layers 4 to 7 plus output, weights" in lines[1]
    assert "RAM: est" in lines[2] and "(no layers, weights" in lines[2]


def test_render_lines_name_the_host_layers_and_output():
    r = _report(**{"n-gpu-layers": 5})
    lines = mf.render_lines(r)
    assert "(layers 4 to 7 plus output, weights" in lines[0]
    assert "(layers 0 to 3, weights" in lines[1]
    none = mf.render_lines(_report(**{"n-gpu-layers": 0}))
    assert "(no layers, weights" in none[0]
    assert "(layers 0 to 7 plus output, weights" in none[1]


def test_render_lines_row_split_names_no_range():
    """Row split lists the layers every card shares rather than naming a
    per-card range, since each card holds a fraction of each layer."""
    r = _report(
        free=(16 * GIB, 8 * GIB), **{"split-mode": "row", "tensor-split": "50,50"}
    )
    lines = mf.render_lines(r)
    assert "(layers 0 to 7, row split" in lines[0]
    assert "(layers 0 to 7, row split" in lines[1]
    assert "plus output" in lines[0] and "plus output" not in lines[1]


def test_render_lines_row_split_partial_offload():
    """A partial offload under row split names the layers actually on a
    card, shared by every card, with the rest on the RAM line."""
    r = _report(
        free=(16 * GIB, 8 * GIB),
        **{"split-mode": "row", "tensor-split": "50,50", "n-gpu-layers": 4},
    )
    lines = mf.render_lines(r)
    assert "(layers 5 to 7, row split" in lines[0]
    assert "(layers 5 to 7, row split" in lines[1]
    assert "(layers 0 to 4, weights" in lines[2]


def test_render_lines_row_split_no_offload():
    """Row split with no layers offloaded names no layers on either card."""
    r = _report(
        free=(16 * GIB, 8 * GIB),
        **{"split-mode": "row", "tensor-split": "50,50", "n-gpu-layers": 0},
    )
    lines = mf.render_lines(r)
    assert "(no layers" in lines[0]
    assert "(no layers" in lines[1]


def test_layer_range_text_joins_split_runs():
    lay = replace(
        _report().estimate.layout,
        card_layers=((0, 1, 2, 3, 8, 9, 10, 11),),
        row_split=False,
        output_device=None,
    )
    assert mf.layer_range_text(lay, 0) == "layers 0 to 3, 8 to 11"
    assert (
        mf.layer_range_text(replace(lay, card_layers=((0, 2, 3, 4),)), 0)
        == "layers 0, 2 to 4"
    )
    assert mf.layer_range_text(replace(lay, card_layers=((),)), 0) == "no layers"
    assert (
        mf.ram_range_text(replace(lay, ram_layers=(4, 5, 6, 7)))
        == "layers 4 to 7 plus output"
    )


def test_render_details_lines():
    r = _report(free=(16 * GIB, 8 * GIB), **{"tensor-split": "40,60"})
    d = mf.render_details(r)
    assert d[0] == "KV per 1024 tokens: 16 MiB (GPU0 8 MiB, GPU1 8 MiB, RAM 0 MiB)"
    assert d[1] == "GPU0: 500 MiB per layer over 4 layers"
    assert d[2] == "GPU1: 500 MiB per layer over 4 layers"
    assert d[3] == "heads 8, KV heads 8, embedding 512, vocabulary 1000"
    assert d[4] == "output tensor 100 MiB on GPU1"
    assert "experts" not in "\n".join(d)


def test_render_details_one_layer_and_unknown_output():
    one = _report(free=(16 * GIB,), **{"n-gpu-layers": 2})
    d = mf.render_details(one)
    assert d[1] == "GPU0: 500 MiB per layer over 1 layer"
    blob = mf.fit_report(
        replace(_meta(), tensors=()),
        8 * GIB,
        settings={"ctx-size": 4096},
        engine="llama.cpp",
        free_bytes_per_gpu=[16 * GIB],
        ram_available=64 * GIB,
        with_details=True,
    )
    blob_lines = mf.render_details(blob)
    assert "weights per layer unknown without a tensor table" in blob_lines
    assert blob_lines[-1] == "output tensor of unknown size on GPU0"
    assert mf.to_json(blob)["cards"][0]["bytes_per_layer"] is None


def test_render_details_moe_and_window_and_ram_output():
    meta = replace(
        _meta(moe=True, swa=1024), sliding_window_pattern=tuple(True for _ in range(8))
    )
    r = _report(meta, **{"n-gpu-layers": 5})
    d = mf.render_details(r)
    assert d[1] == "GPU0: 500 MiB per layer over 4 layers, experts 400 MiB per layer"
    assert d[2].endswith(", sliding window 1024")
    assert d[3] == "output tensor 100 MiB on GPU0"
    none = mf.render_details(_report(**{"n-gpu-layers": 0}))
    assert none[1] == "GPU0: no layers"
    assert none[3] == "output tensor 100 MiB in RAM"


def test_render_details_row_split_shares_the_layer_count():
    """Under row split every card's line uses the shared layer count, not
    its own card_layers, since row split spreads every layer's weight over
    every card."""
    r = _report(
        free=(16 * GIB, 8 * GIB), **{"split-mode": "row", "tensor-split": "50,50"}
    )
    d = mf.render_details(r)
    assert d[1] == "GPU0: 250 MiB per layer over 8 layers, row split"
    assert d[2] == "GPU1: 250 MiB per layer over 8 layers, row split"


def test_details_stay_out_of_tooltip_and_dialog():
    r = _report(free=(3 * GIB,), fit="off")
    assert "per layer" not in mf.render_tooltip(r)
    assert "per layer" not in (mf.render_dialog(r) or "")


def test_to_json_carries_layout_and_kv_per_1k():
    r = _report(free=(16 * GIB, 8 * GIB), **{"tensor-split": "40,60"})
    j = mf.to_json(r)
    assert j["n_layers"] == 8
    assert j["cards"][0]["layers"] == [0, 1, 2, 3]
    assert j["cards"][1]["layers"] == [4, 5, 6, 7]
    assert j["cards"][0]["bytes_per_layer"] == 500 * MIB
    assert j["ram"]["layers"] == []
    assert j["output_device"] == 1
    assert (
        j["kv_per_1k"]["total"] == sum(j["kv_per_1k"]["cards"]) + j["kv_per_1k"]["ram"]
    )
    assert json.loads(json.dumps(j))["n_layers"] == 8
    host = mf.to_json(_report(**{"n-gpu-layers": 0}))
    assert host["output_device"] == "ram" and host["ram"]["layers"] == list(range(8))


def test_to_json_row_split_gives_every_card_the_full_layer_range():
    """Under row split every card's JSON layer list is the same shared
    range, since row split spreads every layer's weight over every card."""
    r = _report(
        free=(16 * GIB, 8 * GIB), **{"split-mode": "row", "tensor-split": "50,50"}
    )
    j = mf.to_json(r)
    assert j["cards"][0]["layers"] == list(range(8))
    assert j["cards"][1]["layers"] == list(range(8))
    assert j["cards"][1]["bytes_per_layer"] == 250 * MIB


def test_fit_report_omits_kv_per_1k_without_details(monkeypatch):
    """fit_report leaves kv_per_1k empty unless with_details is asked for,
    so a caller that only reads the verdict skips the second estimate: one
    estimate_memory call without details, two (the report's own and the
    marginal one) with."""
    calls = []
    real = mf.estimate_memory

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(mf, "estimate_memory", counting)
    r = _report(with_details=False)
    assert r.kv_per_1k == ()
    assert len(calls) == 1
    calls.clear()
    _report(with_details=True)
    assert len(calls) == 2
