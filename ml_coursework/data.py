"""UCI verisini indir, doğrula ve çevrimdışı önbellekten oku."""
from io import BytesIO
from pathlib import Path
import warnings
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EXPECTED_COLUMNS, NUMERIC_COLUMNS, RANDOM_STATE, ROOT, TEST_SIZE

UCI_ZIP = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"


def normalize_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if normalized.isna().any():
        raise ValueError(f"Geçersiz/eksik boolean değer: {series.name}")
    return normalized.astype(bool)


def validate_dataset(frame: pd.DataFrame, expected_rows: int | None = 12330) -> pd.DataFrame:
    if frame.columns.duplicated().any() or set(frame.columns) != set(EXPECTED_COLUMNS):
        raise ValueError("Veri sütunları UCI 468 şemasıyla eşleşmiyor.")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Beklenen satır sayısı {expected_rows}, gelen {len(frame)}.")
    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ["Revenue", "Weekend"]:
        frame[column] = normalize_boolean(frame[column])
    if frame["ProductRelated_Duration"].isna().any():
        raise ValueError("Regresyon hedefi eksik olamaz.")
    values = frame[NUMERIC_COLUMNS].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Sayısal alanlarda negatif/sonsuz değer bulundu.")
    if frame["Revenue"].nunique() != 2:
        raise ValueError("Revenue iki sınıf içermeli.")
    return frame


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path is not None else ROOT / "data" / "online_shoppers_intention.csv"
    if path.exists():
        return validate_dataset(pd.read_csv(path))
    if data_path is not None:
        raise FileNotFoundError(f"Çevrimdışı dosya bulunamadı: {path}")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=468)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        frame = validate_dataset(frame)
    except Exception as error:
        warnings.warn(f"ucimlrepo erişimi başarısız ({type(error).__name__}); resmî UCI ZIP deneniyor.")
        try:
            with urlopen(UCI_ZIP, timeout=60) as response:
                archive = ZipFile(BytesIO(response.read()))
            frame = validate_dataset(pd.read_csv(archive.open("online_shoppers_intention.csv")))
        except Exception as download_error:
            raise RuntimeError(
                "UCI indirilemedi. Resmî CSV'yi data/online_shoppers_intention.csv konumuna koyun "
                "veya --data-path ile belirtin."
            ) from download_error
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def prepare_modeling_data(frame: pd.DataFrame) -> pd.DataFrame:
    # Aynı kayıt train ve testte bulunarak sonucu şişirmesin. Raw EDA değişmez.
    return frame.drop_duplicates().copy()


def shared_split(frame: pd.DataFrame):
    if not frame.index.is_unique:
        raise ValueError("Satır indeksleri benzersiz olmalı.")
    return train_test_split(
        frame.index.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=frame["Revenue"],
    )
