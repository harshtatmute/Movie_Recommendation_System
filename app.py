import pickle
import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Movie Recommender", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# =========================
# LOAD DATA
# =========================
try:
    movies_dict = pickle.load(open(ARTIFACT_DIR / "movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open(ARTIFACT_DIR / "similarity.pkl", "rb"))
except Exception:
    st.error("Artifacts not found. Please generate them first.")
    st.stop()

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(movie):
    try:
        index = movies[movies["title"] == movie].index[0]
    except IndexError:
        return []

    distances = similarity[index]

    movie_scores = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_scores:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# =========================
# UI
# =========================
st.title("🎬 Movie Recommender System")

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)
    else:
        st.warning("No recommendations found.")
