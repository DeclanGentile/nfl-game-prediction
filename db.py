from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
from contextlib import contextmanager
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

def get_engine() -> Engine:
    url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True, future=True)

@contextmanager
def connect():
    eng = get_engine()
    with eng.connect() as conn:
        yield conn

def read_sql_df(query: str, params: dict | None = None) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)
