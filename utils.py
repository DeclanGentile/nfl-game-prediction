import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from config import LOGS_PATH, PROCESSED_DATA_PATH

Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOGS_PATH, f"{name}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def rolling_features(df: pd.DataFrame, group_cols: list[str], sort_cols: list[str],
                     feature_cols: list[str], windows=(3,5)) -> pd.DataFrame:
    """Create rolling means for specified columns without leakage (uses shift)."""
    df = df.sort_values(sort_cols).copy()
    for w in windows:
        for col in feature_cols:
            df[f"{col}_roll{w}"] = (
                df.groupby(group_cols)[col]
                  .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                  .values
            )
    return df

def safe_div(n, d):
    return np.where(d == 0, 0.0, n / d)
