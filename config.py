import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")
DB_NAME = os.getenv("PG_DB")


RAW_DATA_PATH        = os.getenv("RAW_DATA_PATH", "./raw")
PROCESSED_DATA_PATH  = os.getenv("PROCESSED_DATA_PATH", "./processed")
MODELS_PATH          = os.getenv("MODELS_PATH", "./models")
LOGS_PATH            = os.getenv("LOGS_PATH", "./logs")
# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # one level up from scripts/

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")



CURRENT_SEASON = int(os.getenv("CURRENT_SEASON", "2025"))
CURRENT_WEEK   = int(os.getenv("CURRENT_WEEK", "3"))
GAME_TYPES     = os.getenv("GAME_TYPES", "REG").split(",")

TRAIN_SEASONS_START = int(os.getenv("TRAIN_SEASONS_START", "2015"))
TRAIN_SEASONS_END   = int(os.getenv("TRAIN_SEASONS_END", "2021"))
VALID_SEASONS_START = int(os.getenv("VALID_SEASONS_START", "2022"))
VALID_SEASONS_END   = int(os.getenv("VALID_SEASONS_END", "2023"))
TEST_SEASONS_START  = int(os.getenv("TEST_SEASONS_START", "2024"))
TEST_SEASONS_END    = int(os.getenv("TEST_SEASONS_END", "2024"))

SEED = int(os.getenv("SEED", "42"))

# Tables
SCHEMA_CORE = "core"
TABLE_GAMES = f"{SCHEMA_CORE}.games"                # schedule/results with odds
TABLE_PBP   = f"{SCHEMA_CORE}.pbp"                  # condensed pbp 
TABLE_INJ   = f"{SCHEMA_CORE}.injuries_scraped"     # scraper output
