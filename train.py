import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from config import MODELS_PATH, TRAIN_SEASONS_START, TEST_SEASONS_END
from utils import get_logger
from features import build_features

logger = get_logger("train")

def load_game_level_features():
    tg = build_features(
        season_start=TRAIN_SEASONS_START,
        season_end=TEST_SEASONS_END,
        include_future=False
    )
    tg = tg[~tg["team_win"].isna()].copy()

    # Split into home and away rows
    home = tg[tg["is_home"] == 1].copy()
    away = tg[tg["is_home"] == 0].copy()

    # Merge into one row per game
    games = home.merge(
        away,
        on="game_id",
        suffixes=("_home", "_away")
    )

    # Target = did the home team win?
    y = games["team_win_home"].astype(int)

    # Drop known leakage / identifier columns
    exclude = [
        "game_id",
        "season_home", "season_away",
        "week_home", "week_away",
        "kickoff_ts_home", "kickoff_ts_away",
        "home_team_home", "away_team_home",
        "home_score_home", "away_score_home",
        "home_score_away", "away_score_away",
        "team_win_home", "team_win_away",
        "completed_home", "completed_away",
        "season_type_home", "season_type_away",
        "is_home_home", "is_home_away"
    ]

    # Candidate features
    feature_cols = [c for c in games.columns if c not in exclude]

    # Keep only numeric features
    X = games[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
    numeric_features = list(X.columns)

    logger.info(f"[DEBUG] Using {len(numeric_features)} numeric features for training")
    logger.debug(f"Features: {numeric_features}")

    return X, y, numeric_features

def train_and_select_model(X, y):
    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(),
        "xgb": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    param_grid = {
        "logreg": {"C": [0.1, 1, 10]},
        "rf": {"n_estimators": [100, 200], "max_depth": [3, 5, None]},
        "xgb": {"n_estimators": [100, 200], "max_depth": [3, 5]}
    }

    best_model = None
    best_score = float("inf")
    best_name = None

    for name, model in models.items():
        logger.info(f"Tuning {name}…")
        grid = GridSearchCV(model, param_grid[name], scoring="neg_log_loss", cv=3, n_jobs=-1)
        grid.fit(X, y)

        if -grid.best_score_ < best_score:
            best_score = -grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    logger.info(f"Selected best model: {best_name} (logloss={best_score:.4f})")

    # Calibrate probabilities
    logger.info("Calibrating best model…")
    calibrated = CalibratedClassifierCV(best_model, cv=3, method="isotonic")
    calibrated.fit(X, y)

    return calibrated, best_model

def plot_feature_importance(model, features, outdir=MODELS_PATH):
    os.makedirs(outdir, exist_ok=True)
    outpath_img = os.path.join(outdir, "feature_importance.png")
    outpath_csv = os.path.join(outdir, "feature_importance.csv")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not provide feature importances.")
        return

    sorted_idx = np.argsort(importances)[::-1]

    # Save chart
    top_idx = sorted_idx[:20]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(features)[top_idx], importances[top_idx])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.savefig(outpath_img, bbox_inches="tight")
    plt.close()

    # Save CSV
    imp_df = pd.DataFrame({
        "feature": np.array(features)[sorted_idx],
        "importance": importances[sorted_idx]
    })
    imp_df.to_csv(outpath_csv, index=False)

    logger.info(f"Saved feature importance chart → {outpath_img}")
    logger.info(f"Saved feature importance values → {outpath_csv}")

def main():
    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

    logger.info("Loading features…")
    X, y, feature_cols = load_game_level_features()

    logger.info(f"Training on {len(X)} games with {len(feature_cols)} features…")
    model, raw_model = train_and_select_model(X, y)

    # Save model
    model_path = os.path.join(MODELS_PATH, "best_model.joblib")
    joblib.dump(model, model_path)

    # Save features
    feat_file = os.path.join(MODELS_PATH, "best_model_features.txt")
    with open(feat_file, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    # Threshold optimization
    probs = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.49, 0.51, 21)
    best_thresh, best_acc = 0.5, 0
    for t in thresholds:
        acc = (probs >= t).astype(int).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    thresh_file = os.path.join(MODELS_PATH, "best_model_threshold.txt")
    with open(thresh_file, "w") as f:
        f.write(str(best_thresh))

    logger.info(f"Saved model → {model_path}")
    logger.info(f"Saved features → {feat_file}")
    logger.info(f"Saved threshold = {best_thresh:.2f}")

    # Save feature importances (use raw_model, not calibrated wrapper)
    plot_feature_importance(raw_model, feature_cols)

if __name__ == "__main__":
    main()