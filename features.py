import os
import pandas as pd
import numpy as np
from pathlib import Path
from db import read_sql_df
from utils import get_logger, rolling_features, safe_div
from config import (
    TABLE_GAMES, TABLE_PBP, TABLE_INJ,
    PROCESSED_DATA_PATH,
    TRAIN_SEASONS_START, TEST_SEASONS_END
)

"""
features.py — builds model features for training/evaluation and for upcoming-game predictions.

Outputs:
- processed/features_training.parquet  (completed/historical games only)
- processed/features.parquet           (alias of the historical set; used by evaluate/backtest)
- processed/features_full.parquet      (includes future/upcoming games; used by predict)

Changes in this version:
- Adds matchup features (offense vs opponent defense)
- Adds position-weighted injury impact over last 7 days (inj_weighted_last7)
- Fixes ypp_off_vs_def by computing defensive yards/play (def_ypp)
- Prints progress + where files are written when run as a script
"""

logger = get_logger("features")

# Position weights for injuries (simple heuristic)
INJ_WEIGHTS = {
    "QB": 5, "WR": 3, "RB": 2, "TE": 2, "OL": 3,
    "DL": 3, "LB": 2, "DB": 3, "S": 3, "CB": 3
}

# ----------------------
# Loaders
# ----------------------

def load_games(season_start=None, season_end=None, only_upcoming=False) -> pd.DataFrame:
    q = f"""
    SELECT
      g.game_id, g.season, g.week, g.season_type,
      g.kickoff_ts, g.home_team, g.away_team,
      g.home_score, g.away_score,
      g.away_moneyline, g.home_moneyline,
      g.spread_line, g.total_line,
      CASE WHEN g.home_score IS NOT NULL AND g.away_score IS NOT NULL
           THEN (g.home_score > g.away_score) ELSE NULL END AS home_win,
      CASE WHEN g.season_type = 'REG' THEN (g.week IN (1,2,18)) ELSE FALSE END AS early_or_late_reg,
      g.completed
    FROM {TABLE_GAMES} g
    WHERE 1=1
    """
    params = {}
    if season_start is not None and season_end is not None:
        q += " AND g.season BETWEEN :s1 AND :s2"
        params.update({"s1": season_start, "s2": season_end})
    if only_upcoming:
        q += " AND g.completed = FALSE"
    return read_sql_df(q, params)

def load_pbp_agg() -> pd.DataFrame:
    """Aggregate play-by-play to per-game, per-team offense & defense blocks.
    Also compute defensive yards and defensive YPP so matchup YPP features are valid.
    """
    q = f"""
    WITH plays AS (
      SELECT game_id, posteam, defteam, play_type,
             yards_gained::float AS yards_gained,
             epa::float AS epa,
             success::boolean AS success
      FROM {TABLE_PBP}
      WHERE epa IS NOT NULL
    ),
    by_off AS (
      SELECT game_id,
             posteam AS team,
             COUNT(*)                                               AS off_plays,
             SUM(CASE WHEN play_type IN ('pass','run') THEN 1 ELSE 0 END) AS off_rushpass,
             SUM(CASE WHEN play_type='pass' THEN 1 ELSE 0 END)      AS off_pass_plays,
             SUM(yards_gained)                                      AS off_yards,
             AVG(epa)                                               AS off_epa,
             AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)           AS off_success
      FROM plays
      GROUP BY game_id, posteam
    ),
    by_def AS (
      SELECT game_id,
             defteam AS team,
             COUNT(*)                                         AS def_plays,
             SUM(yards_gained)                                AS def_yards,      -- yards allowed
             AVG(epa)                                         AS def_epa,
             AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)     AS def_success
      FROM plays
      GROUP BY game_id, defteam
    )
    SELECT
      COALESCE(o.game_id, d.game_id) AS game_id,
      COALESCE(o.team, d.team)       AS team,
      o.off_plays, o.off_rushpass, o.off_pass_plays, o.off_yards, o.off_epa, o.off_success,
      d.def_plays, d.def_yards, d.def_epa, d.def_success
    FROM by_off o
    FULL OUTER JOIN by_def d
      ON o.game_id = d.game_id AND o.team = d.team;
    """
    return read_sql_df(q)

def load_injuries_agg() -> pd.DataFrame:
    """Daily counts by team & position; compute 7-day window per game later."""
    q = f"""
    SELECT
      team,
      date_trunc('day', date_scraped)::date AS inj_date,
      position,
      COUNT(*) FILTER (WHERE COALESCE(game_status,'') ILIKE '%out%')       AS inj_out,
      COUNT(*) FILTER (WHERE COALESCE(practice_status,'') ILIKE '%Did Not%') AS inj_dnp,
      COUNT(*) FILTER (WHERE COALESCE(practice_status,'') ILIKE '%Limited%')  AS inj_limited,
      COUNT(*) FILTER (WHERE COALESCE(practice_status,'') ILIKE '%Full%')     AS inj_full
    FROM {TABLE_INJ}
    GROUP BY team, inj_date, position;
    """
    return read_sql_df(q)

# ----------------------
# Builders
# ----------------------

def build_team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    home = games.copy()
    home["team"] = home["home_team"]
    home["opp"] = home["away_team"]
    home["is_home"] = 1

    away = games.copy()
    away["team"] = away["away_team"]
    away["opp"] = away["home_team"]
    away["is_home"] = 0

    tg = pd.concat([home, away], ignore_index=True)

    # True team win
    tg["team_win"] = np.where(
        (tg["is_home"] == 1) & (tg["home_win"] == True), 1,
        np.where((tg["is_home"] == 0) & (tg["home_win"] == False), 1,
        np.where(tg["home_win"].isna(), np.nan, 0))
    )

    # Scaled home-field adjustment
    tg["home_field_adj"] = tg["is_home"] - 0.55

    return tg


def attach_pbp(team_games: pd.DataFrame, pbp_agg: pd.DataFrame) -> pd.DataFrame:
    return team_games.merge(pbp_agg, how="left", on=["game_id","team"])

def attach_injuries(team_games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Attach last-7-days injury counts and a position-weighted injury score."""
    if injuries is None or injuries.empty:
        team_games = team_games.copy()
        team_games["num_out_last7"] = 0
        team_games["num_dnp_last7"] = 0
        team_games["num_limited_last7"] = 0
        team_games["num_full_last7"] = 0
        team_games["inj_weighted_last7"] = 0.0
        return team_games

    team_games = team_games.copy()
    team_games["kickoff_date"] = pd.to_datetime(team_games["kickoff_ts"], utc=True).dt.normalize()
    injuries = injuries.copy()
    injuries["inj_date"] = pd.to_datetime(injuries["inj_date"], utc=True).dt.normalize()

    out_list = []
    for _, row in team_games.iterrows():
        kd = row["kickoff_date"]
        team = row["team"]
        window = injuries[
            (injuries["team"] == team)
            & (injuries["inj_date"] >= kd - pd.Timedelta(days=6))
            & (injuries["inj_date"] <= kd)
        ]
        if window.empty:
            out_list.append({
                "num_out_last7": 0, "num_dnp_last7": 0,
                "num_limited_last7": 0, "num_full_last7": 0,
                "inj_weighted_last7": 0.0
            })
            continue

        inj_out = window["inj_out"].sum()
        inj_dnp = window["inj_dnp"].sum()
        inj_limited = window["inj_limited"].sum()
        inj_full = window["inj_full"].sum()

        # Weighted injury score by position importance (sum over the window)
        # window has multiple rows per day/position; aggregate by position then weight
        pos_counts = window.groupby("position", dropna=False)["inj_out"].sum()
        weighted_score = float(sum(INJ_WEIGHTS.get(str(pos).upper(), 1) * cnt for pos, cnt in pos_counts.items()))

        out_list.append({
            "num_out_last7": int(inj_out),
            "num_dnp_last7": int(inj_dnp),
            "num_limited_last7": int(inj_limited),
            "num_full_last7": int(inj_full),
            "inj_weighted_last7": weighted_score
        })

    inj_df = pd.DataFrame(out_list)
    return pd.concat([team_games.reset_index(drop=True), inj_df], axis=1)

def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["off_pass_rate"] = safe_div(df["off_pass_plays"].fillna(0), df["off_rushpass"].fillna(0))
    df["off_ypp"]       = safe_div(df["off_yards"].fillna(0), df["off_plays"].fillna(0))
    # Defensive yards per play allowed
    if "def_yards" in df.columns:
        df["def_ypp"] = safe_div(df["def_yards"].fillna(0), df["def_plays"].fillna(0))
    else:
        df["def_ypp"] = np.nan
    return df

def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["off_epa","off_success","def_epa","def_success","off_ypp","off_pass_rate","def_ypp"]
    return rolling_features(
        df, group_cols=["team"], sort_cols=["season","week","kickoff_ts"],
        feature_cols=base_cols, windows=(3,5)
    )

def add_matchup_features(games: pd.DataFrame) -> pd.DataFrame:
    # Offense vs opponent defense
    games["epa_off_vs_def"] = games["off_epa"] - games["def_epa"]
    games["success_off_vs_def"] = games["off_success"] - games["def_success"]
    games["ypp_off_vs_def"] = games["off_ypp"] - games["def_ypp"]

    # Injury differential (vs average of both teams in the game)
    if "inj_weighted_last7" in games.columns:
        games["inj_diff"] = (
            games["inj_weighted_last7"] -
            games.groupby("game_id")["inj_weighted_last7"].transform("mean")
        )

    # Absolute home vs away differences ---
    # These give the model true team-vs-team comparisons
    for metric in ["off_epa", "def_epa", "off_success", "def_success",
                   "off_ypp", "off_pass_rate", "inj_weighted_last7"]:
        if f"{metric}" in games.columns:
            # Diff = home metric - away metric
            games[f"{metric}_diff"] = (
                games.groupby("game_id")[metric].transform("first") -
                games.groupby("game_id")[metric].transform("last")
            )

    return games


# ----------------------
# Unified feature builder
# ----------------------

def build_features(season_start=None, season_end=None, include_future=False) -> pd.DataFrame:
    logger.info("Loading games…")
    games = load_games(season_start, season_end)
    if not include_future:
        games = games[games["completed"] == True]

    logger.info("Loading PBP aggregates…")
    pbp_agg = load_pbp_agg()

    logger.info("Assembling team-game rows…")
    tg = build_team_game_rows(games)

    logger.info("Merging PBP with team-games…")
    tg = attach_pbp(tg, pbp_agg)

    logger.info("Loading & merging injuries…")
    inj_agg = load_injuries_agg()
    tg = attach_injuries(tg, inj_agg)

    logger.info("Engineering rates…")
    tg = add_rate_features(tg)

    logger.info("Adding rolling form features…")
    tg = add_rolling_form(tg)

    logger.info("Adding matchup features…")
    tg = add_matchup_features(tg)

    tg["kickoff_ts"] = pd.to_datetime(tg["kickoff_ts"], utc=True).dt.tz_convert(None)

    return tg

# ----------------------
# CLI entry
# ----------------------

def main():
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Training set = completed games only
    tg_train = build_features(
        season_start=TRAIN_SEASONS_START,
        season_end=TEST_SEASONS_END,
        include_future=False
    )
    tg_train = tg_train[~tg_train["team_win"].isna()].copy()
    out_train = os.path.join(PROCESSED_DATA_PATH, "features_training.parquet")
    tg_train.to_parquet(out_train, index=False)
    print(f"Wrote {len(tg_train)} rows to {out_train}")

    # Alias for eval/backtest scripts expecting features.parquet
    out_hist = os.path.join(PROCESSED_DATA_PATH, "features.parquet")
    tg_train.to_parquet(out_hist, index=False)
    print(f"Wrote {len(tg_train)} rows to {out_hist}")

    # Full set = includes future games (for predict.py)
    tg_full = build_features(
        season_start=TRAIN_SEASONS_START,
        season_end=TEST_SEASONS_END,
        include_future=True
    )
    out_full = os.path.join(PROCESSED_DATA_PATH, "features_full.parquet")
    tg_full.to_parquet(out_full, index=False)
    print(f"Wrote {len(tg_full)} rows to {out_full}")

    print("Done ✅")

if __name__ == "__main__":
    main()
