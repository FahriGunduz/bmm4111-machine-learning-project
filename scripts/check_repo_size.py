"""Git indeksindeki (yeni stage edilenler dahil) dosya boyutlarını denetle."""
import argparse
from pathlib import Path
import subprocess


def check(root: Path, total_limit: int = 15_000_000, file_limit: int = 2_000_000) -> int:
    entries = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    total = 0
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        _, object_id, stage = metadata.split()
        if stage != "0":
            raise RuntimeError(f"Çözülmemiş Git çatışması: {name}")
        size = int(subprocess.check_output(["git", "cat-file", "-s", object_id], cwd=root))
        total += size
        print(f"{size:>9} bytes  {name}")
        if size > file_limit:
            print(f"UYARI: {name} tek dosya sınırını aşıyor ({file_limit} bytes)")
    print(f"TOTAL: {total} bytes ({total / 1_000_000:.3f} MB); limit={total_limit} bytes")
    return int(total > total_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(check(args.root))
