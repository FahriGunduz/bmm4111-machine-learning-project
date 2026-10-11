import importlib.util
import subprocess

from ml_coursework.config import ROOT


def test_size_checker_reads_staged_blobs_and_enforces_limit(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("check_repo_size", ROOT / "scripts/check_repo_size.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "sample.bin"
    path.write_bytes(b"abc")
    subprocess.run(["git", "add", "sample.bin"], cwd=tmp_path, check=True)
    path.write_bytes(b"working copy is larger but not staged")
    assert module.check(tmp_path, total_limit=3, file_limit=2) == 0
    output = capsys.readouterr().out
    assert "TOTAL: 3 bytes" in output and "UYARI" in output
    assert module.check(tmp_path, total_limit=2) == 1
