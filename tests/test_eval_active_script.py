import json
import subprocess
import sys


def test_eval_active_smoke(tmp_path):
    out_json = tmp_path / "quick_active.json"
    out_md = tmp_path / "quick_active.md"

    cmd = [
        sys.executable,
        "eval_active.py",
        "--config",
        "quick_active_benchmark_config.json",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True)

    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert isinstance(payload, list) and payload
    assert out_md.exists()
