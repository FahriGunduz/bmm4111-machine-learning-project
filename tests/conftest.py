import numpy as np
import pandas as pd
import pytest

from ml_coursework.config import NUMERIC_COLUMNS


@pytest.fixture
def sample_frame():
    rng = np.random.default_rng(42)
    n = 120
    frame = pd.DataFrame({c: rng.uniform(0, 10, n) for c in NUMERIC_COLUMNS})
    frame["Month"] = rng.choice(["May", "Nov"], n)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        frame[c] = rng.integers(1, 4, n)
    frame["VisitorType"] = rng.choice(["New_Visitor", "Returning_Visitor"], n)
    frame["Weekend"] = rng.choice([True, False], n)
    frame["Revenue"] = np.arange(n) % 5 == 0
    return frame
