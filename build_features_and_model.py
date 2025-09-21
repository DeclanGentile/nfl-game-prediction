import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

# ---------------------------
# Config
# ---------------------------
load_dotenv()
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5433")
PG_DB = os.getenv("PG_DB", "nfl_data")

ENGINE = create_engine(f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}")

SEASONS = list(range(2015, 2026))  # 2015..2025
GAME_TYPES = ("REG",)              # start with regular season
ROLL_WINDOWS = [4, 8]              # last-4, last-8 games

# Map long names from scraper -> abbreviations used in core.games
TEAM_MAP = {
    # AFC East
    "Bills": "BUF", "Dolphins": "MIA", "Patriots": "NE", "Jets": "NYJ",
    # AFC North
    "Ravens": "BAL", "Bengals": "CIN", "Browns": "CLE", "Steelers": "PIT",
    # AFC South
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Titans": "TEN",
    # AFC West
    "Broncos": "DEN", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC",
    # NFC East
    "Cowboys": "DAL", "Giants": "NYG", "Eagles": "PHI", "Commanders": "WAS",
    # NFC North
    "Bears": "CHI", "Lions": "DET", "Packers": "GB", "Vikings": "MIN",
    # NFC South
    "Falcons": "ATL", "Panthers": "CAR", "Saints": "NO", "Buccaneers": "TB",
    # NFC West
    "Cardinals": "ARI", "Rams": "LAR", "49ers": "SF", "Seahawks": "SEA",
    # Safety nets
    "Miami Dolphins": "MIA", "New York Jets": "NYJ", "New York Giants": "NYG",
    "San Francisco 49ers": "SF", "Los Angeles Rams": "LAR",
    "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV",
}

POS_GROUPS = {
    "OL": {"T","G","C"},
    "DB": {"CB","S"},
    "WR": {"WR"},
    "RB": {"RB"},
    "TE": {"TE"},
    "QB": {"QB"},
}

# ---------------------------
# Helpers
# ---------------------------
def read_games():
    q = """
        SELECT 
          game_id, season, week, season_type, kickoff_ts,
          home_team, away_team, home_score, away_score, completed
        FROM core.games
        WHERE season = ANY(:seasons)
          AND season_type = ANY(:game_types)
    """
    df = pd.read_sql(text(q), ENGINE, params={"seasons": SEASONS, "game_types": list(GAME_TYPES)})
    df["kickoff_ts"] = pd.to_datetime(df["kickoff_ts"], errors="coerce", utc=True).dt.tz_convert(None)
    return df

def read_pbp():
    # We only need a few columns; if your pbp table is huge, consider indexing (you did) and selecting only needed cols
    q = """
        SELECT game_id, play_id, qtr, down, ydstogo, yardline_100, posteam, defteam, play_type, yards_gained, epa, success
        FROM core.pbp
        WHERE game_id IS NOT NULL
    """
    df = pd.read_sql(text(q), ENGINE)
    return df

def team_game_stats_from_pbp(pbp):
    # Aggregate posteam-by-game: EPA/play and success rate
    # Filter out plays with missing epa / success where appropriate
    df = pbp.copy()
    df = df[df["posteam"].notna()]
    grp = df.groupby(["game_id","posteam"], as_index=False).agg(
        plays=("epa","size"),
        epa_per_play=("epa","mean"),
        success_rate=("success", "mean"),
    )
    grp = grp.rename(columns={"posteam":"team"})
    return grp

def add_schedule_context(games):
    # Rest days: difference between this kickoff and team's previous game kickoff
    # Build long frame with one row per (game, team as home/away)
    home = games[["game_id","season","week","season_type","kickoff_ts","home_team"]].copy()
    home["team"] = home["home_team"]; home["is_home"]=1
    away = games[["game_id","season","week","season_type","kickoff_ts","away_team"]].copy()
    away["team"] = away["away_team"]; away["is_home"]=0
    ga = pd.concat([
        home.rename(columns={"home_team":"opp_team"})[["game_id","season","week","season_type","kickoff_ts","team","is_home"]],
        away.rename(columns={"away_team":"opp_team"})[["game_id","season","week","season_type","kickoff_ts","team","is_home"]],
    ], ignore_index=True)

    ga = ga.sort_values(["team","kickoff_ts"])
    ga["prev_kick"] = ga.groupby("team")["kickoff_ts"].shift(1)
    ga["rest_days"] = (ga["kickoff_ts"] - ga["prev_kick"]).dt.days
    ga["short_week"] = (ga["rest_days"].fillna(99) <= 6).astype(int)
    ga["off_bye"] = (ga["rest_days"].fillna(0) >= 13).astype(int)  # rough bye proxy (≥ 13 days)
    return ga[["game_id","team","is_home","rest_days","short_week","off_bye"]]

def read_injuries_snapshot():
    # Scraped table you created
    q = """
        SELECT team, player, position, injury, practice_status, game_status, date_scraped
        FROM core.injuries_scraped
    """
    inj = pd.read_sql(text(q), ENGINE)
    if inj.empty:
        return inj
    inj["date_scraped"] = pd.to_datetime(inj["date_scraped"], utc=True, errors="coerce").dt.tz_convert(None)
    # Normalize team to abbreviations used in core.games
    inj["team_std"] = inj["team"].map(TEAM_MAP).fillna(inj["team"])
    # Flags
    inj["is_out"] = (inj["game_status"].fillna("").str.strip().str.lower() == "out").astype(int)
    inj["is_questionable"] = (inj["game_status"].fillna("").str.strip().str.lower() == "questionable").astype(int)
    inj["pos_clean"] = inj["position"].fillna("").str.upper().str.strip()
    # Position-group specific OUT flags
    for gname, gset in POS_GROUPS.items():
        inj[f"out_{gname.lower()}"] = inj.apply(lambda r: 1 if (r["is_out"]==1 and r["pos_clean"] in gset) else 0, axis=1)
    return inj

def pick_injury_snapshot_for_game(inj, team, kickoff):
    # Choose the latest snapshot at or before kickoff for this team
    sub = inj[(inj["team_std"]==team) & (inj["date_scraped"]<=kickoff)]
    if sub.empty:
        return { "num_out":0, "num_questionable":0, **{f"out_{g.lower()}":0 for g in POS_GROUPS} }
    latest_ts = sub["date_scraped"].max()
    latest = sub[sub["date_scraped"]==latest_ts]
    out_counts = latest["is_out"].sum()
    q_counts = latest["is_questionable"].sum()
    feats = { "num_out": int(out_counts), "num_questionable": int(q_counts) }
    for gname in POS_GROUPS:
        feats[f"out_{gname.lower()}"] = int(latest[f"out_{gname.lower()}"].sum())
    return feats

def rolling_team_form(team_games, windows=(4,8)):
    # team_games has one row per (game_id, team) with epa_per_play, success_rate, kickoff_ts
    team_games = team_games.sort_values(["team","kickoff_ts"]).copy()
    for w in windows:
        # shift(1) so current game does NOT include itself -> leakage-safe
        team_games[f"epa_pp_l{w}"] = team_games.groupby("team")["epa_per_play"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        team_games[f"sr_l{w}"] = team_games.groupby("team")["success_rate"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    return team_games

def make_game_feature_row(row, team_long, side_prefix):
    # Extract per-team features for home/away merge step
    out = {
        f"{side_prefix}_is_home": int(row["is_home"]),
        f"{side_prefix}_rest_days": row["rest_days"] if pd.notna(row["rest_days"]) else 7,
        f"{side_prefix}_short_week": int(row["short_week"]),
        f"{side_prefix}_off_bye": int(row["off_bye"]),
        f"{side_prefix}_epa_pp_l4": row.get("epa_pp_l4", np.nan),
        f"{side_prefix}_epa_pp_l8": row.get("epa_pp_l8", np.nan),
        f"{side_prefix}_sr_l4": row.get("sr_l4", np.nan),
        f"{side_prefix}_sr_l8": row.get("sr_l8", np.nan),
        f"{side_prefix}_num_out": row.get("num_out", 0),
        f"{side_prefix}_num_questionable": row.get("num_questionable", 0),
        f"{side_prefix}_out_ol": row.get("out_ol", 0),
        f"{side_prefix}_out_db": row.get("out_db", 0),
        f"{side_prefix}_out_wr": row.get("out_wr", 0),
        f"{side_prefix}_out_rb": row.get("out_rb", 0),
        f"{side_prefix}_out_te": row.get("out_te", 0),
        f"{side_prefix}_out_qb": row.get("out_qb", 0),
    }
    return out

# ---------------------------
# Build features
# ---------------------------
print("Loading games…")
games = read_games()
games = games[games["completed"].fillna(True)]  # keep completed (historical training)
print(f"Games loaded: {len(games)}")

print("Loading PBP and building team game stats…")
pbp = read_pbp()
tgs = team_game_stats_from_pbp(pbp)  # one row per (game_id, team)
# attach kickoff to tgs for sorting/rolling
tgs = tgs.merge(games[["game_id","kickoff_ts"]], on="game_id", how="left")

# Compute rolling form (last-4 and last-8) for each team
tgs_roll = rolling_team_form(tgs, windows=ROLL_WINDOWS)

# Schedule context per team-game
print("Building schedule context…")
sched_ctx = add_schedule_context(games)

# Merge rolling stats + context -> team-game frame
team_game = (
    tgs_roll.merge(sched_ctx, on=["game_id","team"], how="left")
             .merge(games[["game_id","season","week","home_team","away_team","kickoff_ts","home_score","away_score"]],
                    on="game_id", how="left")
)

# Injuries snapshot
print("Loading injuries snapshots…")
inj = read_injuries_snapshot()

# Compose game-level dataset with home/away blocks
rows = []
for _, g in games.iterrows():
    gid = g["game_id"]; kickoff = g["kickoff_ts"]; ht = g["home_team"]; at = g["away_team"]

    # pull per-team rows
    hrow = team_game[(team_game["game_id"]==gid) & (team_game["team"]==ht)].iloc[0] if not team_game[(team_game["game_id"]==gid) & (team_game["team"]==ht)].empty else None
    arow = team_game[(team_game["game_id"]==gid) & (team_game["team"]==at)].iloc[0] if not team_game[(team_game["game_id"]==gid) & (team_game["team"]==at)].empty else None

    # injury snapshot (most recent <= kickoff)
    if inj is not None and not inj.empty:
        hinj = pick_injury_snapshot_for_game(inj, ht, kickoff)
        ainj = pick_injury_snapshot_for_game(inj, at, kickoff)
    else:
        hinj = { "num_out":0, "num_questionable":0, **{f"out_{g.lower()}":0 for g in POS_GROUPS} }
        ainj = { "num_out":0, "num_questionable":0, **{f"out_{g.lower()}":0 for g in POS_GROUPS} }

    base = {
        "game_id": gid, "season": g["season"], "week": g["week"],
        "kickoff_ts": kickoff, "home_team": ht, "away_team": at,
        "home_score": g["home_score"], "away_score": g["away_score"],
        "home_win": 1 if (pd.notna(g["home_score"]) and pd.notna(g["away_score"]) and g["home_score"] > g["away_score"]) else np.nan
    }

    # if stats missing (rare early seasons), fill NaNs
    if hrow is None:
        hfeat = {k: np.nan for k in ["epa_pp_l4","epa_pp_l8","sr_l4","sr_l8"]}
        hctx = {"is_home":1, "rest_days":7, "short_week":0, "off_bye":0}
        hblock = {**{ "epa_pp_l4":np.nan, "epa_pp_l8":np.nan, "sr_l4":np.nan, "sr_l8":np.nan }, **hctx}
    else:
        hblock = hrow

    if arow is None:
        ablock = {**{ "epa_pp_l4":np.nan, "epa_pp_l8":np.nan, "sr_l4":np.nan, "sr_l8":np.nan },
                  **{"is_home":0, "rest_days":7, "short_week":0, "off_bye":0}}
    else:
        ablock = arow

    # assemble per-side dicts
    home_feats = make_game_feature_row(hblock, ht, "home")
    away_feats = make_game_feature_row(ablock, at, "away")

    # append injuries
    for k,v in hinj.items():
        home_feats[f"home_{k}"] = v
    for k,v in ainj.items():
        away_feats[f"away_{k}"] = v

    # construct diff features (home - away)
    diffs = {}
    for k in list(home_feats.keys()):
        if k.startswith("home_"):
            kk = k.replace("home_","")
            ak = f"away_{kk}"
            if ak in away_feats:
                diffs[f"diff_{kk}"] = (
                    (home_feats[k] if pd.notna(home_feats[k]) else 0) -
                    (away_feats[ak] if pd.notna(away_feats[ak]) else 0)
                )

    rows.append({**base, **home_feats, **away_feats, **diffs})

df = pd.DataFrame(rows)

# Basic cleaning: drop rows without outcome (shouldn’t be many in historical)
df = df[df["home_win"].notna()].copy()
feature_cols = [c for c in df.columns if c.startswith("diff_")] + [
    "home_is_home","home_rest_days","home_short_week","home_off_bye",
    "away_is_home","away_rest_days","away_short_week","away_off_bye",
]

# You can also include the raw per-side features (not just diffs):
feature_cols += [
    "home_epa_pp_l4","home_epa_pp_l8","home_sr_l4","home_sr_l8",
    "away_epa_pp_l4","away_epa_pp_l8","away_sr_l4","away_sr_l8",
    "home_num_out","home_num_questionable","home_out_ol","home_out_db","home_out_wr","home_out_rb","home_out_te","home_out_qb",
    "away_num_out","away_num_questionable","away_out_ol","away_out_db","away_out_wr","away_out_rb","away_out_te","away_out_qb",
]

# Fill NaNs conservatively
df[feature_cols] = df[feature_cols].fillna(0)

# Time-aware split
train_mask = df["season"].between(2015, 2023)
valid_mask = (df["season"]==2024)
test_mask  = (df["season"]==2025)

X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "home_win"].astype(int)
X_valid, y_valid = df.loc[valid_mask, feature_cols], df.loc[valid_mask, "home_win"].astype(int)
X_test,  y_test  = df.loc[test_mask,  feature_cols], df.loc[test_mask,  "home_win"].astype(int)

print(f"Train games: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# ---------------------------
# Baseline model: Logistic Regression
# ---------------------------
logit = LogisticRegression(max_iter=2000)
logit.fit(X_train, y_train)

def eval_model(name, model, X, y):
    p = model.predict_proba(X)[:,1]
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(y, yhat)
    ll  = log_loss(y, p, labels=[0,1])
    bs  = brier_score_loss(y, p)
    auc = roc_auc_score(y, p)
    print(f"{name} -> Acc: {acc:.3f} | LogLoss: {ll:.3f} | Brier: {bs:.3f} | AUC: {auc:.3f}")

print("\nLogistic Regression performance:")
eval_model("Train", logit, X_train, y_train)
if len(X_valid): eval_model("Valid", logit, X_valid, y_valid)
if len(X_test):  eval_model("Test",  logit, X_test,  y_test)

# ---------------------------
# Non-linear model: Gradient Boosting
# ---------------------------
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

print("\nGradient Boosting performance:")
eval_model("Train", gb, X_train, y_train)
if len(X_valid): eval_model("Valid", gb, X_valid, y_valid)
if len(X_test):  eval_model("Test",  gb, X_test,  y_test)

# Optional: write features to DB for audit / later modeling
try:
    outcols = ["game_id","season","week","home_team","away_team","home_score","away_score","home_win"] + feature_cols
    df[outcols].to_sql("model_features", ENGINE, schema="core", if_exists="replace", index=False)
    print("\nSaved core.model_features for inspection.")
except Exception as e:
    print(f"Note: could not save model_features to DB: {e}")
