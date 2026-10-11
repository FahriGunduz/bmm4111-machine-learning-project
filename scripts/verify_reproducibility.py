"""Aynı kaynaktan iki çalıştırmanın sayısal/metinsel tablolarını karşılaştır."""
import argparse
import json
from pathlib import Path

import pandas as pd


def compare(expected: Path, actual: Path):
    count = 0
    for path in sorted((expected / "tables").glob("*.csv")):
        other = actual / "tables" / path.name
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(other), check_exact=False, rtol=1e-10, atol=1e-8)
        count += 1
    for path in sorted((expected / "tables").glob("*.json")):
        other = actual / "tables" / path.name
        first = json.loads(path.read_text(encoding="utf-8"))
        second = json.loads(other.read_text(encoding="utf-8"))
        assert first == second, f"JSON farklı: {path.name}"
        count += 1
    print(f"PASS: {count} sonuç tablosu/JSON yeniden üretildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    compare(args.expected, args.actual)
