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


def _report(meta=None, free=(16 * GIB,), ram=64 * GIB, engine="llama.cpp", **settings):
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


def test_offload_count_stays_at_the_profile_split_when_balanced_fits_alone():
    """When the balanced split fits on its own, the offload count search
    still runs at the profile's own split: the message names the count
    found there and never claims no offload count fits."""
    meta = _meta(n_layers=16)
    kw = dict(
        settings={"ctx-size": 8192, "n-gpu-layers": "all"},
        engine="llama.cpp",
        free_bytes_per_gpu=[5 * GIB, 5 * GIB],
        ram_available=64 * GIB,
    )
    r = mf.fit_report(meta, 0, **kw)
    assert r.balanced is not None and r.balanced.fits is True
    at_profile = mf.smallest_fitting_offload(
        meta,
        0,
        settings=kw["settings"],
        engine="llama.cpp",
        free_bytes_per_gpu=kw["free_bytes_per_gpu"],
    )
    assert at_profile is not None
    key, value = at_profile
    text = " ".join(m.text for m in r.messages)
    assert f"--{key} {value}" in text
    assert "no offload count fits" not in text


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
