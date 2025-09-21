import os
import pandas as pd
import streamlit as st
import pytz

# Paths
MODELS_PATH = "H:/NFL/models"
LOGO_DIR = "H:/NFL/logos"

# Load predictions
csv_path = os.path.join(MODELS_PATH, "predictions_upcoming.csv")
df = pd.read_csv(csv_path)

# Only show the next week’s games
next_week = df["week"].min()
df = df[df["week"] == next_week].copy()

st.set_page_config(page_title="NFL Game Predictions", layout="wide")
st.title(f"NFL Predictions - Week {next_week}")

# Sort by confidence (highest first)
df = df.sort_values("confidence", ascending=False)

# Timezones
utc = pytz.utc
est = pytz.timezone("US/Eastern")

# Card style wrapper
card_style = """
<div style="
    border: 1px solid #ddd; 
    border-radius: 6px; 
    padding: 6px; 
    margin-bottom: 6px;
    background-color: #fafafa;
    text-align: center;
    font-size: 14px;
">
    {content}
</div>
"""

# Loop in chunks of 4 games
for i in range(0, len(df), 4):
    cols = st.columns(4)

    for j, (_, row) in enumerate(df.iloc[i:i+4].iterrows()):
        kickoff = pd.to_datetime(row["kickoff_ts"], utc=True).astimezone(est)
        kickoff_str = kickoff.strftime("%a %b %d, %I:%M %p EST")
        conf_pct = row["confidence"] * 100

        home_logo = os.path.join(LOGO_DIR, f"{row['home_team']}.png")
        away_logo = os.path.join(LOGO_DIR, f"{row['away_team']}.png")

        with cols[j]:
            # Date text (cleaner, no white box)
            st.markdown(f"""
            <div style="color: gray; font-size: 13px; margin-bottom: 4px;">
                {kickoff_str}
            </div>
            """, unsafe_allow_html=True)

            # Teams + logos
            c1, c2, c3 = st.columns([1, 0.5, 1])
            with c1:
                st.image(away_logo, width=40)
                st.write(f"{row['away_team']} {'✅' if row['pick'] == row['away_team'] else ''}")
                st.write(f"{row['away_win_prob']*100:.1f}%")
            with c2:
                st.write("vs")
            with c3:
                st.image(home_logo, width=40)
                st.write(f"{row['home_team']} {'✅' if row['pick'] == row['home_team'] else ''}")
                st.write(f"{row['home_win_prob']*100:.1f}%")

            # Confidence bar
            st.caption(f"Confidence: {conf_pct:.1f}%")
            st.progress(min(1.0, conf_pct / 100))
