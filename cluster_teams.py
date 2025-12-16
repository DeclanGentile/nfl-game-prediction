import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# If you don't have a config.py, you can comment this out and set paths manually below
try:
    from config import PROCESSED_DATA_PATH, SEED
except ImportError:
    PROCESSED_DATA_PATH = "."
    SEED = 42

# -------- SETTINGS --------
K_OFF = 4
K_DEF = 4

# Logo / Visual Settings (Matches Script 1)
LOGOS_DIR = "logos"
TARGET_LOGO_HEIGHT_PX = 16
TRIM_TRANSPARENT_BORDERS = True
HULL_ALPHA = 0.15
HULL_EDGE_ALPHA = 0.7

PROFILES_FILE = os.path.join(PROCESSED_DATA_PATH, "team_profiles.csv")

OUT_OFF_CSV = os.path.join(PROCESSED_DATA_PATH, "team_clusters_offense_latest.csv")
OUT_DEF_CSV = os.path.join(PROCESSED_DATA_PATH, "team_clusters_defense_latest.csv")

OUT_OFF_SUM = os.path.join(PROCESSED_DATA_PATH, "team_cluster_summary_offense_latest.csv")
OUT_DEF_SUM = os.path.join(PROCESSED_DATA_PATH, "team_cluster_summary_defense_latest.csv")

OUT_OFF_PNG = os.path.join(PROCESSED_DATA_PATH, "team_clusters_offense_latest.png")
OUT_DEF_PNG = os.path.join(PROCESSED_DATA_PATH, "team_clusters_defense_latest.png")
# -------------------------


def _add_logo(ax, xy, logo_path):
    """Helper to add a team logo at x,y coordinates."""
    try:
        img = Image.open(logo_path).convert("RGBA")
        if TRIM_TRANSPARENT_BORDERS:
            alpha = img.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                img = img.crop(bbox)
        w, h = img.size
        if h != TARGET_LOGO_HEIGHT_PX:
            new_w = int(round(w * (TARGET_LOGO_HEIGHT_PX / h)))
            img = img.resize((new_w, TARGET_LOGO_HEIGHT_PX), Image.LANCZOS)
        arr = np.asarray(img)
        oi = OffsetImage(arr, zoom=1.0)
        ab = AnnotationBbox(oi, xy, frameon=False, zorder=3)
        ax.add_artist(ab)
        return True
    except Exception:
        return False


def _draw_cluster_hull(ax, x_vals, y_vals, color):
    """Draw a translucent convex hull around the (x,y) points using scipy."""
    pts = np.column_stack([x_vals, y_vals])
    if pts.shape[0] < 3:
        return
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                facecolor=color, alpha=HULL_ALPHA, edgecolor=color,
                linewidth=1.5, zorder=0)
        ax.plot(np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color=color, alpha=HULL_EDGE_ALPHA, linewidth=1.5, zorder=1)
    except Exception:
        pass


def pca_axis_labels(pca: PCA, feature_cols: list[str], top_n: int = 3) -> tuple[str, str]:
    """Label axes using the top contributing original features."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1", "PC2"]
    )

    def label(pc):
        top = loadings[pc].abs().sort_values(ascending=False).head(top_n).index.tolist()
        return f"{pc} (top): " + ", ".join(top)

    return label("PC1"), label("PC2")


def cluster_and_plot(df: pd.DataFrame, id_cols: list[str], feature_cols: list[str],
                     k: int, out_csv: str, out_summary: str, out_png: str, title: str) -> None:
    """Fit KMeans, save cluster CSV + summary, and create a 2D PCA plot with logos and hulls."""
    
    # 1. Clustering
    X = df[feature_cols].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=SEED, n_init=25)
    labels = km.fit_predict(Xs)

    # Save mapping
    out_map = df[id_cols].copy()
    out_map["cluster"] = labels
    out_map.to_csv(out_csv, index=False)

    # Save summary means
    tmp = df.copy()
    tmp["cluster"] = labels
    summary = tmp.groupby("cluster")[feature_cols].mean().reset_index()
    summary.to_csv(out_summary, index=False)

    # 2. PCA for visualization
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(Xs)
    tmp["viz_x"] = Z[:, 0]
    tmp["viz_y"] = Z[:, 1]

    xlab, ylab = pca_axis_labels(pca, feature_cols, top_n=3)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger for better logo spacing
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    clusters = sorted(tmp["cluster"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(clusters)}
    
    # A. Draw Hulls and Background Points
    x_vals = tmp["viz_x"].values
    y_vals = tmp["viz_y"].values

    for c in clusters:
        m = (tmp["cluster"] == c)
        # Draw Hull
        _draw_cluster_hull(ax, x_vals[m], y_vals[m], colors[c])
        # Draw faint scatter points (background for logos)
        ax.scatter(x_vals[m], y_vals[m], color=colors[c], alpha=0.18, s=60, zorder=2, label=f"Cluster {c}")

    # B. Draw Logos (or fallback dots)
    # Determine which column holds the team name for logo lookup
    team_col = None
    for col in ["team", "posteam"]:
        if col in tmp.columns:
            team_col = col
            break

    for _, row in tmp.iterrows():
        xv, yv = row["viz_x"], row["viz_y"]
        cluster_id = row["cluster"]
        
        team_name = row[team_col] if team_col else None
        logo_file = os.path.join(LOGOS_DIR, f"{team_name}.png") if team_name else None
        
        # Try to add logo; if it fails, draw a solid dot
        ok = bool(team_name) and os.path.isfile(logo_file) and _add_logo(ax, (xv, yv), logo_file)
        
        if not ok:
            # Fallback dot
            ax.scatter([xv], [yv], s=80, edgecolor="k", linewidth=0.5, alpha=0.9,
                       color=colors[cluster_id], zorder=3)
            # Add text annotation if no logo
            if team_name:
                 ax.annotate(str(team_name), (xv, yv), xytext=(4, 2), 
                             textcoords="offset points", fontsize=8, alpha=0.8, zorder=4)

    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"✅ Saved: {out_csv}")
    print(f"✅ Saved: {out_summary}")
    print(f"✅ Saved: {out_png}")


def main():
    if not os.path.exists(PROFILES_FILE):
        raise FileNotFoundError(f"Missing {PROFILES_FILE}. Run your team profile builder first.")

    df = pd.read_csv(PROFILES_FILE)

    # Identify columns
    if "season" not in df.columns:
        raise ValueError("team_profiles.csv must include a 'season' column for latest-season filtering.")
    if "team" in df.columns:
        team_col = "team"
    elif "posteam" in df.columns:
        team_col = "posteam"
    else:
        raise ValueError("team_profiles.csv must include 'team' or 'posteam'.")

    latest_season = int(pd.to_numeric(df["season"], errors="coerce").dropna().max())
    df = df[df["season"] == latest_season].copy()

    id_cols = ["season", team_col]

    # Pick offense + defense feature sets by prefix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove IDs that may be numeric (season)
    numeric_cols = [c for c in numeric_cols if c not in ["season"]]

    off_cols = [c for c in numeric_cols if c.startswith("off_")]
    def_cols = [c for c in numeric_cols if c.startswith("def_")]

    # Safety removals
    for bad in ["games", "week", "is_home", "home_field_adj", "team_win", "target"]:
        if bad in off_cols: off_cols.remove(bad)
        if bad in def_cols: def_cols.remove(bad)

    if len(off_cols) < 2:
        raise ValueError(f"Not enough offense features found. Found {len(off_cols)} columns starting with 'off_'.")
    if len(def_cols) < 2:
        raise ValueError(f"Not enough defense features found. Found {len(def_cols)} columns starting with 'def_'.")

    # Fill any missing values in selected sets
    df[off_cols] = df[off_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df[def_cols] = df[def_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Latest season detected: {latest_season}")
    print(f"Offense features: {len(off_cols)} | Defense features: {len(def_cols)}")

    cluster_and_plot(
        df=df[id_cols + off_cols],
        id_cols=id_cols,
        feature_cols=off_cols,
        k=K_OFF,
        out_csv=OUT_OFF_CSV,
        out_summary=OUT_OFF_SUM,
        out_png=OUT_OFF_PNG,
        title=f"Offense Style Clusters (K={K_OFF}) — Season {latest_season}"
    )

    cluster_and_plot(
        df=df[id_cols + def_cols],
        id_cols=id_cols,
        feature_cols=def_cols,
        k=K_DEF,
        out_csv=OUT_DEF_CSV,
        out_summary=OUT_DEF_SUM,
        out_png=OUT_DEF_PNG,
        title=f"Defense Style Clusters (K={K_DEF}) — Season {latest_season}"
    )


if __name__ == "__main__":
    main()