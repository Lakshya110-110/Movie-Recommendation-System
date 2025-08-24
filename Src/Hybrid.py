import pandas as pd
from Src.Data import load_movies, load_ratings
from Src.Collaborative import collaborative_recommend
from Src.Content import content_based_recommend

def hybrid_recommend(user_id, movie_title, top_n=10):
    # Get recommendations from both methods
    collab_recs = collaborative_recommend(user_id, top_n*2)
    content_recs = content_based_recommend(movie_title, top_n*2)

    # Merge results (movies appearing in both lists get boosted)
    merged = pd.merge(collab_recs, content_recs, on="title", how="outer", suffixes=("_collab", "_content"))

    # Combine scores
    merged["final_score"] = merged[["score_collab", "score_content"]].mean(axis=1)

    return merged.sort_values("final_score", ascending=False).head(top_n)[["title", "final_score"]]

if __name__ == "__main__":
    print(hybrid_recommend(1, "Toy Story (1995)", 10))
