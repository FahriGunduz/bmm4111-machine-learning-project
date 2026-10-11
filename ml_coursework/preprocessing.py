"""Bütün görevlerde hedefleri dışlayan ortak, fold içinde öğrenilen dönüşümler."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def task_features(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError(f"Bilinmeyen görev: {task}")
    # Allowlist yeni hedef türevlerinin fark edilmeden modele girmesini önler.
    return frame.loc[:, FEATURE_COLUMNS].copy()


def build_preprocessor(scale: bool = True, drop_first: bool = True) -> ColumnTransformer:
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" if drop_first else None)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=True)


def build_pipeline(estimator, scale: bool = True, drop_first: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(scale, drop_first)), ("model", estimator)])
