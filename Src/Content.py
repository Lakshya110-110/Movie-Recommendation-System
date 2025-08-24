import pandas as pd
from Src.Data import load_movies, load_ratings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommend(movie_title, top_n=10):
    movies = load_movies()

    #TF-IDF on genres
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

    #find index of input movie
    idx = movies[movies["title"].str.lower() == movie_title.lower()].index[0]

    #compute similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    #get top recommendations
    sim_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return movies.iloc[sim_indices][["title", "genres"]].assign(score=cosine_sim[sim_indices])

if __name__ == "__main__":
    print(content_based_recommend("Toy Story (1995)", 10))