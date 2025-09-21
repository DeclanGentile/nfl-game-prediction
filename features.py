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
             COUNT(*) AS off_plays,
             SUM(CASE WHEN play_type IN ('pass','run') THEN 1 ELSE 0 END) AS off_rushpass,
             SUM(CASE WHEN play_type='pass' THEN 1 ELSE 0 END) AS off_pass_plays,
             SUM(yards_gained) AS off_yards,
             AVG(epa) AS off_epa,
             AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) AS off_success
      FROM plays
      GROUP BY game_id, posteam
    ),
    by_def AS (
      SELECT game_id,
             defteam AS team,
             COUNT(*) AS def_plays,
             SUM(yards_gained) AS def_yards,
             AVG(epa) AS def_epa,
             AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) AS def_success
      FROM plays
      GROUP BY game_id, defteam
    )
    SELECT
      COALESCE(o.game_id, d.game_id) AS game_id,
      COALESCE(o.team, d.team) AS team,
      o.off_plays, o.off_rushpass, o.off_pass_plays, o.off_yards, o.off_epa, o.off_success,
      d.def_plays, d.def_yards, d.def_epa, d.def_success
    FROM by_off o
    FULL OUTER JOIN by_def d
      ON o.game_id = d.game_id AND o.team = d.team;
    """
    return read_sql_df(q)

def load_injuries_agg() -> pd.DataFrame:
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

    # Label: did the team win?
    tg["team_win"] = np.where(
        (tg["is_home"] == 1) & (tg["home_win"] == True), 1,
        np.where((tg["is_home"] == 0) & (tg["home_win"] == False), 1,
        np.where(tg["home_win"].isna(), np.nan, 0))
    )

    # Home field adjustment
    tg["home_field_adj"] = tg["is_home"] - 0.55

    # Drop leakage columns
    tg = tg.drop(columns=["home_score", "away_score", "home_win"])

    return tg

def attach_pbp(team_games: pd.DataFrame, pbp_agg: pd.DataFrame) -> pd.DataFrame:
    return team_games.merge(pbp_agg, how="left", on=["game_id","team"])

def attach_injuries(team_games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    if injuries is None or injuries.empty:
        for col in ["num_out_last7","num_dnp_last7","num_limited_last7","num_full_last7","inj_weighted_last7"]:
            team_games[col] = 0.0
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
            out_list.append({"num_out_last7": 0, "num_dnp_last7": 0,
                             "num_limited_last7": 0, "num_full_last7": 0,
                             "inj_weighted_last7": 0.0})
            continue

        inj_out = window["inj_out"].sum()
        inj_dnp = window["inj_dnp"].sum()
        inj_limited = window["inj_limited"].sum()
        inj_full = window["inj_full"].sum()

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
    df["def_ypp"]       = safe_div(df["def_yards"].fillna(0), df["def_plays"].fillna(0))
    return df

def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["off_epa","off_success","def_epa","def_success","off_ypp","off_pass_rate","def_ypp"]
    # shift(1) ensures only *past* games are included
    for col in base_cols:
        df[col] = df[col].shift(1)
    return rolling_features(
        df, group_cols=["team"], sort_cols=["season","week","kickoff_ts"],
        feature_cols=base_cols, windows=(3,5)
    )

def add_matchup_features(games: pd.DataFrame) -> pd.DataFrame:
    games["epa_off_vs_def"] = games["off_epa"] - games["def_epa"]
    games["success_off_vs_def"] = games["off_success"] - games["def_success"]
    games["ypp_off_vs_def"] = games["off_ypp"] - games["def_ypp"]
    return games

# ----------------------
# Unified builder
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
# CLI
# ----------------------

def main():
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)

    tg_train = build_features(TRAIN_SEASONS_START, TEST_SEASONS_END, include_future=False)
    tg_train = tg_train[~tg_train["team_win"].isna()].copy()
    tg_train.to_parquet(os.path.join(PROCESSED_DATA_PATH, "features_training.parquet"), index=False)
    tg_train.to_parquet(os.path.join(PROCESSED_DATA_PATH, "features.parquet"), index=False)

    tg_full = build_features(TRAIN_SEASONS_START, TEST_SEASONS_END, include_future=True)
    tg_full.to_parquet(os.path.join(PROCESSED_DATA_PATH, "features_full.parquet"), index=False)

    print("Wrote training, features, and full datasets ✅")

if __name__ == "__main__":
    main()
