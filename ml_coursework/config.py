"""Deney ayarları ve semantik sütun türleri."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
NUMERIC_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_COLUMNS = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
]
EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["Revenue"]
EXCLUDED_COLUMNS = {"Revenue", "ProductRelated_Duration", "PageValues"}
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in EXCLUDED_COLUMNS]


def output_dirs(output_dir: Path):
    tables, figures = Path(output_dir) / "tables", Path(output_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures
