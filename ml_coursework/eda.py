"""Ham verinin şeffaf kalite ve dağılım incelemesi; model ayarı seçmez."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, output_dirs
from .evaluation import save_figure, save_json


def run_eda(frame: pd.DataFrame, output_dir: Path):
    tables, figures = output_dirs(output_dir)
    summary = {
        "rows": len(frame), "columns": len(frame.columns), "original_features": len(frame.columns) - 1,
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "numeric_columns": NUMERIC_COLUMNS, "categorical_columns": CATEGORICAL_COLUMNS,
        "missing_counts": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "modeling_rows_after_exact_deduplication": len(frame.drop_duplicates()),
        "revenue_counts": frame.Revenue.value_counts().rename(index={True: "True", False: "False"}).to_dict(),
        "duration_summary": frame.ProductRelated_Duration.describe().to_dict(),
        "zero_duration_rows": int(frame.ProductRelated_Duration.eq(0).sum()),
        "categorical_values": {c: [str(v) for v in frame[c].dropna().unique()] for c in CATEGORICAL_COLUMNS},
        "source": "https://doi.org/10.24432/C5F88Q", "license": "CC BY 4.0",
    }
    save_json(summary, tables / "dataset_summary.json")
    frame[NUMERIC_COLUMNS].describe().T.to_csv(tables / "numeric_summary.csv")
    pd.DataFrame({"count": frame.isna().sum(), "ratio": frame.isna().mean()}).to_csv(tables / "missing_values.csv")
    # Gözlem örnekleri konsolda incelenir; ham kayıtlar Git'e gönderilmez.
    print(frame.head(3).to_string(index=False), flush=True)
    print(frame.tail(3).to_string(index=False), flush=True)
    print(f"RAW={frame.shape}; duplicates={summary['duplicate_rows']}; missing={frame.isna().sum().sum()}", flush=True)
    outliers = []
    for column in NUMERIC_COLUMNS:
        q1, q3 = frame[column].quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        outliers.append({"feature": column, "q1": q1, "q3": q3, "lower": lower, "upper": upper,
                         "outlier_count": count, "ratio": count / len(frame), "iqr_zero": bool(iqr == 0)})
    pd.DataFrame(outliers).to_csv(tables / "outlier_summary.csv", index=False)
    plt.figure(figsize=(5, 4))
    frame.Revenue.value_counts().sort_index().plot.bar()
    plt.xticks([0, 1], ["Satın almadı", "Satın aldı"], rotation=0)
    plt.ylabel("Oturum sayısı")
    save_figure(figures / "class_distribution.png")
    plt.figure(figsize=(10, 4))
    frame.isna().sum().plot.bar()
    plt.ylabel("Eksik değer sayısı")
    save_figure(figures / "missing_values.png")
    frame[NUMERIC_COLUMNS].hist(figsize=(13, 10), bins=40)
    save_figure(figures / "numeric_distributions.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(frame[NUMERIC_COLUMNS].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    save_figure(figures / "correlation.png")
    fig, axes = plt.subplots(2, 5, figsize=(14, 7))
    for column, ax in zip(NUMERIC_COLUMNS, axes.flat):
        sns.boxplot(y=frame[column], ax=ax)
        ax.set_title(column, fontsize=9)
        ax.set_ylabel("")
    save_figure(figures / "outliers.png")
    return summary
