from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "Movie_Recommendation_System"
AUTHOR_USER_NAME = "harshtatmute"   
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = [
    "numpy",
    "pandas",
    "scikit-learn",
    "streamlit"
]


setup(
    name="movie-recommender-system",
    version="0.1.0",
   author=AUTHOR_USER_NAME,
   description="Content-based Movie Recommendation System using cosine similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
