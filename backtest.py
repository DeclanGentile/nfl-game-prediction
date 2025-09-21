import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from config import MODELS_PATH, TRAIN_SEASONS_START, TEST_SEASONS_END
from utils import get_logger
from features import build_features

logger = get_logger("backtest")


def main():
    # --- Load trained model and feature list ---
    model = joblib.load(os.path.join(MODELS_PATH, "best_model.joblib"))

    feat_file = os.path.join(MODELS_PATH, "best_model_features.txt")
    with open(feat_file) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # --- Build features just like train.py ---
    df = build_features(
        season_start=TRAIN_SEASONS_START,
        season_end=TEST_SEASONS_END,
        include_future=False
    )
    df = df[~df["team_win"].isna()].copy()

    # Merge into game-level rows (home + away)
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()
    games = home.merge(away, on="game_id", suffixes=("_home", "_away"))

    # Target = did the home team win?
    y = games["team_win_home"].astype(int)

    results = []

    for yr in sorted(games["season_home"].unique()):
        sub = games[games["season_home"] == yr]
        if sub.empty:
            continue

        # Keep only trained features (numeric only, same as train/predict)
        cols = [c for c in feature_cols if c in sub.columns]
        X = sub[cols].select_dtypes(include=[np.number]).fillna(0.0)

        # Predict probabilities (home win prob)
        p = model.predict_proba(X)[:, 1]

        # Season metrics
        res = {
            "season": yr,
            "games": len(sub),
            "logloss": log_loss(y.loc[sub.index], p),
            "auc": roc_auc_score(y.loc[sub.index], p),
            "acc_game": ((p >= 0.5).astype(int) == y.loc[sub.index]).mean()
        }

        results.append(res)
        logger.info(f"{yr}: {res}")

    # Save results
    out = pd.DataFrame(results)
    outpath = os.path.join(MODELS_PATH, "backtest_by_season.csv")
    out.to_csv(outpath, index=False)
    logger.info(f"Wrote {outpath} ✅")


if __name__ == "__main__":
    main()
