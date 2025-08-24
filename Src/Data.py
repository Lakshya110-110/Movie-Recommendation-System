from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "movies.csv")
    df["genres"] = df["genres"].fillna("").str.replace("I", "", regex=False)
    return df

def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ratings.csv")

def load_tags() -> pd.DataFrame:
    p = DATA_DIR / "tags.csv"
    if not p.exists():
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    return pd.csv(p)

def build_tags_corpus() -> pd.DataFrame:
    tags = load_tags()
    if tags.empty:
        return pd.DataFrame(columns=["movieId", "tags_text"])
    agg = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(map(str, s.unique())))
    return agg.reset_index().rename(columns={"tag": "tags_text"})
