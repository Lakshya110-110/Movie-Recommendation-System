import streamlit as st
from Src.popularity import top_popular
from Src.Collaborative import collaborative_recommend
from Src.Data import load_ratings

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# Sidebar for navigation
method = st.sidebar.selectbox("Choose Recommendation Method", ["Popularity-based", "Collaborative Filtering"])

if method == "Popularity-based":
    st.header("🔥 Popular Movies")
    n = st.slider("Number of movies to recommend", 5, 20, 10)
    min_ratings = st.slider("Minimum ratings required", 50, 500, 100)
    results = top_popular(n=n, min_ratings=min_ratings)
    st.dataframe(results)

elif method == "Collaborative Filtering":
    st.header("👥 Personalized Recommendations")

    # Load ratings once
    ratings = load_ratings()

    # Select a valid user from the dataset
    valid_users = ratings["userId"].unique()
    user_id = st.selectbox("Select User ID", valid_users)

    n = st.slider("Number of movies to recommend", 5, 20, 10)

    if st.button("Get Recommendations"):
        results = collaborative_recommend(user_id=user_id, top_n=n, ratings=ratings)
        if results.empty:
            st.warning("⚠️ No recommendations found for this user. Try another user ID.")
        else:
            st.dataframe(results)

