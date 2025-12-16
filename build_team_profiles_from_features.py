import os
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH, CURRENT_SEASON

FEATURES_FILE = os.path.join(PROCESSED_DATA_PATH, "features_training.parquet")
OUT_FILE = os.path.join(PROCESSED_DATA_PATH, "team_profiles.csv")

# Do NOT include outcomes / odds / lines in clustering features
DROP_COLS_EXACT = {
    "team_win", "target", "result",
    "home_score", "away_score",
    "total", "spread_line", "total_line",
    "home_moneyline", "away_moneyline",
}

DROP_IF_CONTAINS = [
    "score", "result", "win", "moneyline", "spread", "odds", "total_line"
]

MIN_GAMES = 4


def main():
    print("Building team profiles from FEATURES...")

    df = pd.read_parquet(FEATURES_FILE)

    # Required identifiers
    if "team" not in df.columns:
        raise ValueError("Expected a 'team' column in features_training.parquet")

    group_cols = ["team"]
    if "season" in df.columns:
        group_cols = ["season", "team"]

    # Numeric features
    numeric = (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
          .copy()
    )

    # ✅ Critical fix: remove grouping cols from numeric to prevent overlap (season/team)
    numeric = numeric.drop(columns=[c for c in group_cols if c in numeric.columns], errors="ignore")

    # Drop leakage columns by exact name
    numeric = numeric.drop(columns=[c for c in DROP_COLS_EXACT if c in numeric.columns], errors="ignore")

    # Drop leakage columns by substring
    to_drop = [c for c in numeric.columns if any(s in c.lower() for s in DROP_IF_CONTAINS)]
    numeric = numeric.drop(columns=to_drop, errors="ignore")

    # Build profiles (mean of numeric columns per team-season)
    profiles = numeric.join(df[group_cols]).groupby(group_cols).mean().reset_index()

    # Add games count for stability + filter small samples
    counts = df.groupby(group_cols).size().reset_index(name="games")
    profiles = profiles.merge(counts, on=group_cols, how="left")
    profiles = profiles[profiles["games"] >= MIN_GAMES].copy()

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    profiles.to_csv(OUT_FILE, index=False)

    print(f"✅ Saved team profiles → {OUT_FILE}")
    print(f"Rows: {len(profiles)} | Features used: {profiles.shape[1] - len(group_cols) - 1} (excluding 'games')")


if __name__ == "__main__":
    main()
