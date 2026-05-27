# Data-Driven Movie & Theater Recommendation System

**Author:** Vaseem Ahamed  
**GitHub:** [github.com/vaseemahamedtech](https://github.com/vaseemahamedtech)  
**Stack:** Python, Pandas, NumPy, Scikit-learn, Folium, Matplotlib, Geopy, A* Algorithm

---

## 🚀 Overview

A comprehensive movie and theater recommendation application featuring two core pipelines:
1. **Content-Based Movie Recommendation:** Uses TF-IDF, cosine similarity, A\* genre-graph pathfinding, and interactive Folium geospatial mapping.
2. **Location-Based Theater Recommendation:** Uses A\* Heuristic Search to recommend the best local theater based on user distance, ticket prices, movie ratings, and genre matching, with Matplotlib visual analytics.

---

## 📁 Project Structure

```
movie_recommendation_system/
├── main.py                 # Orchestrates the CLI interactive menu & pipelines
├── data_loader.py          # Data generation, cleaning & feature engineering
├── eda.py                  # Exploratory data analysis & statistics
├── recommender.py          # TF-IDF + cosine similarity movie recommender
├── astar.py                # A* pathfinding on genre graph
├── visualise.py            # Folium interactive HTML map generator
├── theater_recommender.py  # Location-based theater recommender (A* heuristic)
├── requirements.txt        # Package dependencies
├── .gitignore              # Version control ignore rules
├── data/
│   └── movies.csv          # Auto-generated synthetic movie metadata
└── output/
    ├── movie_map.html      # Interactive geospatial movie origin map
    ├── theater_map.png     # Scatter plot of recommended vs local theaters
    └── astar_costs.png     # Bar chart of A* heuristic cost calculations
```

---

## ⚙️ Pipelines & Architecture

### Pipeline 1: Content-Based Movie Recommendation
```
Raw CSV ➔ Clean & Deduplicate ➔ EDA ➔ TF-IDF Recommender ➔ A* Genre Path ➔ Folium HTML Map
```

### Pipeline 2: Location-Based Theater Recommendation
```
User Location ➔ Geocoding (Geopy/Fallback) ➔ A* Heuristic Costing ➔ Priority Queue ➔ Matplotlib Visuals
```

---

## 🛠️ Key Components

### 1. Data Cleaning (`data_loader.py`)
- Deduplication via `drop_duplicates()` (~60 duplicate rows removed).
- Imputes missing rating and votes using the median.
- Feature engineering: `primary_genre`, `profit_usd`, `roi`, `decade`, and TF-IDF `soup` combining title, genre, and director.

### 2. TF-IDF Recommender (`recommender.py`)
- Fits TF-IDF vectorizer on movie metadata `soup` using bigrams.
- Evaluates similarity with cosine similarity.
- Validates quality with Spearman rank correlation between content-similarity order and user-rating order.

### 3. A* Genre Pathfinding (`astar.py`)
- Builds a graph where nodes represent genres and edges represent inverse average ratings.
- Runs A\* search to discover the highest-quality thematic path from a starting genre cluster to a target genre cluster.

### 4. Folium Mapping (`visualise.py`)
- Renders an interactive dark-themed map in HTML.
- Clusters markers color-coded by IMDb ratings (Green: $\ge 7.5$, Yellow: $5.5\text{–}7.4$, Red: $< 5.5$).
- Places a HeatMap overlay visualizing production density and origins.

### 5. A* Location-Based Theater Recommender (`theater_recommender.py`)
- Geocodes user location queries using Geopy (Nominatim API) with retry logic and coordinate fallback.
- Scores theater options based on an A* Heuristic cost:
$$\text{Cost} = \text{Distance (km)} + \frac{\text{Ticket Price}}{100} + (5 - \text{Rating}) + \text{Genre Penalty}$$
  - *Genre Penalty*: Adds $+15$ cost if the movie playing does not match your preferred genre.
- Generates 2D scatter plots of theaters and horizontal bar charts ranking the A\* heuristic costs.

---

## 🚀 Usage

### Installation
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run the App
Launch the interactive CLI menu:
```bash
python main.py
```

If you are running in a headless shell or do not want the graphical Matplotlib windows to block execution, run with the `--no-show` flag to write the plots straight to files:
```bash
python main.py --no-show
```

---

## 📝 Methodology Notes

- **A\* Theater Recommendation Cost:** Minimizing the heuristic cost dynamically balances travel distance, ticket affordability, movie quality, and genre alignment.
- **Spearman Rank Correlation ($\rho$):** Calculated on top recommendations to verify if similarity-based scores correlate with general audience rating statistics.
