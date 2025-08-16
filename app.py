from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")

# Features we want to use for recommendations
def combine_features(row):
    return str(row['genres']) + " " + str(row['keywords']) + " " + str(row['cast']) + " " + str(row['director'])

movies["combined"] = movies.apply(combine_features, axis=1)

# Convert text to vectors
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies["combined"])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(count_matrix)

# Function to recommend movies
def get_recommendations(title):
    if title not in movies['title'].values:
        return ["Movie not found in dataset!"]
    idx = movies[movies.title == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[1:6]  # top 5 recommendations
    movie_indices = [i[0] for i in sorted_scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form["movie_name"]
        recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
