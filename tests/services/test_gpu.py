import llama_launcher.services.gpu as gpu
from llama_launcher.services.gpu import parse_nvidia_smi


def test_parse_nvidia_smi():
    text = "8192, 24576, 16384, 37, 55, NVIDIA GeForce RTX 4090\n"
    rows = parse_nvidia_smi(text)
    assert len(rows) == 1
    r = rows[0]
    assert r.mem_used_mib == 8192 and r.mem_total_mib == 24576
    assert r.mem_free_mib == 16384 and r.util_pct == 37 and r.temp_c == 55
    assert r.name == "NVIDIA GeForce RTX 4090"


def test_parse_handles_na():
    rows = parse_nvidia_smi("[N/A], 24576, [N/A], [Not Supported], 50, GPU0\n")
    assert rows[0].mem_used_mib == 0 and rows[0].util_pct == 0


def test_free_vram_bytes(monkeypatch):
    monkeypatch.setattr(gpu, "query_gpus", lambda: parse_nvidia_smi("100, 200, 150, 10, 40, A\n"))
    assert gpu.free_vram_bytes() == 150 * 1024 * 1024


def test_free_vram_none_when_no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "query_gpus", lambda: [])
    assert gpu.free_vram_bytes() is None
