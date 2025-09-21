import os

# path to your logos folder
logos_dir = r"H:/NFL/logos"

# map of keywords to team abbreviations
team_map = {
    "cardinals": "ARI", "arizona": "ARI",
    "falcons": "ATL", "atlanta": "ATL",
    "ravens": "BAL", "baltimore": "BAL",
    "bills": "BUF", "buffalo": "BUF",
    "panthers": "CAR", "carolina": "CAR",
    "bears": "CHI", "chicago": "CHI",
    "bengals": "CIN", "cincinnati": "CIN",
    "browns": "CLE", "cleveland": "CLE",
    "cowboys": "DAL", "dallas": "DAL",
    "broncos": "DEN", "denver": "DEN",
    "lions": "DET", "detroit": "DET",
    "packers": "GB", "green-bay": "GB", "greenbay": "GB",
    "texans": "HOU", "houston": "HOU",
    "colts": "IND", "indianapolis": "IND",
    "jaguars": "JAX", "jacksonville": "JAX",
    "chiefs": "KC", "kansas-city": "KC", "kansascity": "KC",
    "raiders": "LV", "las-vegas": "LV", "oakland": "LV",
    "chargers": "LAC", "los-angeles-chargers": "LAC", "san-diego": "LAC",
    "rams": "LA", "los-angeles-rams": "LA", "st-louis": "LA",
    "dolphins": "MIA", "miami": "MIA",
    "vikings": "MIN", "minnesota": "MIN",
    "patriots": "NE", "new-england": "NE",
    "saints": "NO", "new-orleans": "NO",
    "giants": "NYG", "new-york-giants": "NYG",
    "jets": "NYJ", "new-york-jets": "NYJ",
    "eagles": "PHI", "philadelphia": "PHI",
    "steelers": "PIT", "pittsburgh": "PIT",
    "49ers": "SF", "san-francisco": "SF",
    "seahawks": "SEA", "seattle": "SEA",
    "buccaneers": "TB", "tampa-bay": "TB",
    "titans": "TEN", "tennessee": "TEN",
    "commanders": "WAS", "washington": "WAS", "redskins": "WAS"
}

for filename in os.listdir(logos_dir):
    lower = filename.lower()
    matched = None
    for keyword, abbr in team_map.items():
        if keyword in lower:
            matched = abbr
            break
    if matched:
        old_path = os.path.join(logos_dir, filename)
        new_path = os.path.join(logos_dir, f"{matched}.png")
        os.rename(old_path, new_path)
        print(f"Renamed {filename} → {matched}.png")
