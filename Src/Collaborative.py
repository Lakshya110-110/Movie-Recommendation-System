from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_recommend(user_id, ratings, top_n=10):
    # Map users & movies to indices
    user_cat = pd.Categorical(ratings["userId"])
    movie_cat = pd.Categorical(ratings["movieId"])

    ratings["user_index"] = user_cat.codes
    ratings["movie_index"] = movie_cat.codes

    # Build sparse user-movie matrix
    user_movie_matrix = csr_matrix(
        (ratings["rating"], (ratings["user_index"], ratings["movie_index"])),
        shape=(len(user_cat.categories), len(movie_cat.categories))
    )

    # Compute user similarity (sparse-aware)
    user_similarity = cosine_similarity(user_movie_matrix)

    # Get index of this user
    try:
        user_index = list(user_cat.categories).index(user_id)
    except ValueError:
        return pd.DataFrame(columns=["title", "score"])

    # Similar users
    sim_scores = list(enumerate(user_similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Pick top similar users
    top_users = [i for i, score in sim_scores[1:50]]

    # Movies rated by top similar users
    similar_users_ratings = user_movie_matrix[top_users].mean(axis=0).A1

    # Recommend top movies not seen by this user
    user_seen = user_movie_matrix[user_index].toarray().flatten() > 0
    scores = [
        (movie_cat.categories[i], score)
        for i, score in enumerate(similar_users_ratings)
        if not user_seen[i]
    ]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    return pd.DataFrame(scores, columns=["movieId", "score"])



