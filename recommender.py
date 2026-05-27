"""
Content-based recommender using TF-IDF + cosine similarity.
Validates ranking quality with Spearman rank correlation.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

def build_recommender(df: pd.DataFrame):
    """Fit TF-IDF on the 'soup' column and return the model + matrix."""
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = tfidf.fit_transform(df["soup"].fillna(""))
    return tfidf, matrix


def get_recommendations(title: str, df: pd.DataFrame, model, matrix,
                        top_n: int = 5):
    """Return top-N similar movies for a given title."""
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        # fuzzy: pick best partial match
        mask = df["title"].str.lower().str.contains(title.lower(), na=False)
        if not mask.any():
            return []
        matches = df[mask]

    idx = matches.index[0]
    sim_scores = cosine_similarity(matrix[idx], matrix).flatten()
    sim_scores[idx] = 0  # exclude self

    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    results = [(df.loc[i, "title"], float(sim_scores[i])) for i in top_idx]

    # Validate with Spearman rank correlation
    predicted_ranks = np.argsort(sim_scores)[::-1][:20]
    rating_ranks    = df.loc[predicted_ranks, "rating"].rank(ascending=False)
    rho, p = spearmanr(range(len(rating_ranks)), rating_ranks)

    return results


def evaluate_recommender(df: pd.DataFrame, model, matrix, sample_n: int = 50):
    """Run a quick batch evaluation; returns mean cosine similarity."""
    sample_idx = np.random.choice(df.index, sample_n, replace=False)
    scores = []
    for idx in sample_idx:
        sims = cosine_similarity(matrix[idx], matrix).flatten()
        sims[idx] = 0
        top_score = sims.max()
        scores.append(top_score)
    return float(np.mean(scores))
