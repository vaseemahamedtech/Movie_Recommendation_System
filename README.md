# Data-Driven Movie Recommendation System

**Author:** Vaseem Ahamed  
**GitHub:** [github.com/vaseemahamedtech](https://github.com/vaseemahamedtech)  
**Stack:** Python, Pandas, NumPy, Scikit-learn, Folium, A* Algorithm

---

## Overview

A content-based movie recommendation system with geospatial visualisation and A* genre-graph pathfinding.

## Project Structure

```
movie_recommendation_system/
├── main.py              # Orchestrates the full pipeline
├── data_loader.py       # Data generation, cleaning & feature engineering
├── eda.py               # Exploratory data analysis & statistics
├── recommender.py       # TF-IDF + cosine similarity recommender
├── astar.py             # A* pathfinding on genre graph
├── visualise.py         # Folium interactive map generator
├── data/
│   └── movies.csv       # Auto-generated synthetic dataset
├── output/
│   └── movie_map.html   # Interactive geospatial output
└── requirements.txt
```

## Pipeline

```
Raw CSV → Clean & Deduplicate → EDA → TF-IDF Recommender → A* Genre Path → Folium Map
```

## Key Components

### 1. Data Cleaning (`data_loader.py`)
- Deduplication via `drop_duplicates()` (~60 rows removed)
- Missing rating/votes imputed with median
- Feature engineering: `primary_genre`, `profit_usd`, `roi`, `decade`, TF-IDF `soup`

### 2. EDA (`eda.py`)
- Rating distribution statistics
- Genre and decade frequency counts
- Correlation matrix (rating, votes, runtime, revenue, budget)

### 3. Recommender (`recommender.py`)
- TF-IDF vectorisation on `title + genre + director` with bigrams
- Cosine similarity ranking
- Spearman rank correlation for quality validation

### 4. A* Pathfinding (`astar.py`)
- Genre-relationship graph where edges represent content proximity
- A* traversal finds minimum-cost genre path (e.g. Action → Animation)
- Useful for generating thematic recommendation trails

### 5. Geospatial Map (`visualise.py`)
- Folium `MarkerCluster` with colour-coded rating circles
- `HeatMap` overlay for production density
- Interactive popup with title, genre, rating, votes

## Usage

```bash
pip install -r requirements.txt
python main.py
# Open output/movie_map.html in a browser
```

## Methodology Notes

- **Similarity Metric:** Cosine similarity on TF-IDF vectors outperforms raw count vectors by normalising for document length, reducing bias toward verbose entries.
- **A* Application:** Genre graph edges are weighted by inverse rating (`10 - avg_rating`), so paths through high-quality genre clusters have lower cost — surfacing recommendation trails aligned with user satisfaction.
- **Validation:** Spearman ρ between cosine-similarity rank and user-rating rank quantifies whether the similarity ordering aligns with crowd opinion.
