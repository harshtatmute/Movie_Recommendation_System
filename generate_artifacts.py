import pandas as pd
import pickle
import ast
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Load Datasets
# -----------------------------
movies = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
credits = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")


# -----------------------------
# 3. Merge Datasets
# -----------------------------
movies = movies.merge(credits, on="title")


# -----------------------------
# 4. Select Required Columns
# -----------------------------
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]


# -----------------------------
# 5. Handle Missing Values
# -----------------------------
movies.dropna(inplace=True)


# -----------------------------
# 6. Convert JSON Columns
# -----------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# -----------------------------
# 7. Extract Top 3 Cast
# -----------------------------
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


movies['cast'] = movies['cast'].apply(convert_cast)


# -----------------------------
# 8. Extract Director
# -----------------------------
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)


# -----------------------------
# 9. Clean Text (Remove Spaces)
# -----------------------------
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# -----------------------------
# 10. Create Tags Column
# -----------------------------
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# -----------------------------
# 11. Vectorization
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


# -----------------------------
# 12. Cosine Similarity
# -----------------------------
similarity = cosine_similarity(vectors)


# -----------------------------
# 13. Save Artifacts
# -----------------------------
pickle.dump(new_df, open(ARTIFACT_DIR / "movie_list.pkl", "wb"))
pickle.dump(similarity, open(ARTIFACT_DIR / "similarity.pkl", "wb"))

print("Artifacts generated successfully!")
