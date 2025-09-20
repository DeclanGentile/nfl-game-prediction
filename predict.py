import os
import joblib
import numpy as np
import pandas as pd
from config import MODELS_PATH, CURRENT_SEASON
from features import build_features
from utils import get_logger

logger = get_logger("predict")


# ----------------------
# Load model + metadata
# ----------------------
def load_model():
    path = os.path.join(MODELS_PATH, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = joblib.load(path)

    # load features separately
    features_file = os.path.join(MODELS_PATH, "best_model_features.txt")
    with open(features_file, "r") as f:
        features = [line.strip() for line in f if line.strip()]

    # load threshold separately
    threshold_file = os.path.join(MODELS_PATH, "best_model_threshold.txt")
    with open(threshold_file, "r") as f:
        threshold = float(f.read().strip())

    logger.info(f"Loaded model from {path}")
    return model, features, threshold


# ----------------------
# Build upcoming games frame (game-level)
# ----------------------
def upcoming_game_features():
    # Get full feature set for current season
    tg = build_features(
        season_start=CURRENT_SEASON,
        season_end=CURRENT_SEASON,
        include_future=True
    )

    # Only upcoming (no result yet)
    tg_upcoming = tg[tg["team_win"].isna()].copy()
    if tg_upcoming.empty:
        return pd.DataFrame()

    # Split into home/away rows
    home = tg_upcoming[tg_upcoming["is_home"] == 1].copy()
    away = tg_upcoming[tg_upcoming["is_home"] == 0].copy()

    # Merge into one row per game
    games = home.merge(
        away,
        on="game_id",
        suffixes=("_home", "_away")
    )

    # Keep identifying info
    out = games[[
        "game_id", "season_home", "week_home",
        "home_team_home", "away_team_home", "kickoff_ts_home"
    ]].copy()
    out = out.rename(columns={
        "season_home": "season",
        "week_home": "week",
        "home_team_home": "home_team",
        "away_team_home": "away_team",
        "kickoff_ts_home": "kickoff_ts"
    })

    return out, games


# ----------------------
# Main
# ----------------------
def main():
    model, features, threshold = load_model()

    logger.info("Building features for upcoming games…")
    out, games = upcoming_game_features()
    if games.empty:
        print("No upcoming games found.")
        return

    # Filter to numeric columns only, in the order model expects
    cols = [c for c in features if c in games.columns]
    X = games[cols].select_dtypes(include=[np.number]).fillna(0.0).astype(float)
    
    # Probabilities (P(home team wins))
    proba_home = model.predict_proba(X)[:, 1]
    proba_away = 1 - proba_home

    out["home_win_prob"] = proba_home
    out["away_win_prob"] = proba_away
    out["pick"] = np.where(proba_home >= threshold, out["home_team"], out["away_team"])
    out["confidence"] = np.abs(proba_home - 0.5) * 2  # 0..1

    # Clean human-readable string
    out["prediction_str"] = (
        out["home_team"] + " " + (out["home_win_prob"] * 100).round(1).astype(str) + "% vs "
        + out["away_team"] + " " + (out["away_win_prob"] * 100).round(1).astype(str)
    )

    # Print to console
    print("\n=== Upcoming Game Predictions (Team vs Team) ===")
    for _, row in out.sort_values(["season", "week", "kickoff_ts"]).iterrows():
        print(f"{row['game_id']} | {row['prediction_str']} | Pick: {row['pick']} | Conf: {row['confidence']:.2f}")

    # Save to CSV
    csv_path = os.path.join(MODELS_PATH, "predictions_upcoming.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nSaved predictions → {csv_path}")


if __name__ == "__main__":
    main()
