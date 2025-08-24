import pandas as pd
from Src.Data import load_movies, load_ratings

def top_popular(n: int = 10, min_ratings: int = 100) -> pd.DataFrame:
    # Load data
    movies = load_movies()
    ratings = load_ratings()

    # Aggregate ratings
    agg = ratings.groupby("movieId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    )

    # Filter movies with at least min_ratings and sort
    cand = agg[agg["num_ratings"] >= min_ratings].sort_values(
        ["avg_rating", "num_ratings"],
        ascending=False
    )

    # Merge with movies metadata
    return cand.head(n).merge(
        movies, on="movieId", how="left"
    )[
        ["movieId", "title", "genres", "avg_rating", "num_ratings"]
    ]


# For testing directly
if __name__ == "__main__":
    print(top_popular(10, 100))
