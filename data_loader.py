"""
Data loading and cleaning pipeline.
Handles missing values, deduplication, type casting, and feature engineering.
"""

import pandas as pd
import numpy as np
import os, random
from datetime import datetime

GENRES   = ["Action", "Adventure", "Animation", "Comedy", "Crime",
            "Drama", "Fantasy", "Family", "Mystery", "Romance",
            "Sci-Fi", "Thriller"]
LANGUAGES = ["English", "French", "Spanish", "Japanese", "Korean",
             "Hindi", "German", "Italian"]
COUNTRIES = {
    "English":  ("United States",   37.09, -95.71),
    "French":   ("France",          46.23,   2.21),
    "Spanish":  ("Spain",           40.46,  -3.75),
    "Japanese": ("Japan",           36.20, 138.25),
    "Korean":   ("South Korea",     35.90, 127.77),
    "Hindi":    ("India",           20.59,  78.96),
    "German":   ("Germany",         51.17,  10.45),
    "Italian":  ("Italy",           41.87,  12.57),
}

def _generate_synthetic_movies(n=1500, seed=42):
    """Synthesise a realistic movie metadata dataset."""
    random.seed(seed); np.random.seed(seed)
    words = ["Dark","Lost","Rising","Fallen","Broken","Last","First","Hidden",
             "Shadow","Light","Storm","Fire","Ice","Wind","Star","Night",
             "Dawn","Dusk","Echo","Void","Ghost","Dream","Eternal","Crimson"]
    nouns = ["Knight","City","World","Journey","Legacy","Chronicle","Protocol",
             "Empire","Horizon","Signal","Code","Blade","Mind","Gate","Hour"]

    rows = []
    for i in range(n):
        lang    = random.choice(LANGUAGES)
        country, lat, lon = COUNTRIES[lang]
        genre1  = random.choice(GENRES)
        genre2  = random.choice([g for g in GENRES if g != genre1])
        title   = f"The {random.choice(words)} {random.choice(nouns)}" if i % 3 != 0 \
                  else f"{random.choice(words)} {random.choice(nouns)}"
        year    = random.randint(1970, 2023)
        rating  = round(np.random.normal(6.5, 1.5), 1)
        rating  = max(1.0, min(10.0, rating))
        votes   = int(np.random.exponential(50000))
        runtime = random.randint(75, 210)
        revenue = round(np.random.exponential(50) * 1e6, 0)
        budget  = round(revenue * random.uniform(0.3, 2.0), 0)
        director = f"Director_{random.randint(1,200)}"

        # inject noise
        if random.random() < 0.04: rating  = None
        if random.random() < 0.03: votes   = None
        if random.random() < 0.05: revenue = None

        rows.append({
            "title": title, "year": year, "genre": f"{genre1}|{genre2}",
            "language": lang, "country": country,
            "rating": rating, "votes": votes,
            "runtime_min": runtime, "revenue_usd": revenue,
            "budget_usd": budget, "director": director,
            "lat": lat + np.random.normal(0, 2),
            "lon": lon + np.random.normal(0, 3),
        })

    df = pd.DataFrame(rows)
    # add ~60 duplicates
    dup_idx = np.random.choice(df.index, 60, replace=False)
    return pd.concat([df, df.loc[dup_idx]], ignore_index=True)


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load CSV (or generate) then clean."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        df_raw = _generate_synthetic_movies()
        df_raw.to_csv(path, index=False)
    else:
        df_raw = pd.read_csv(path)

    print(f"  Raw records        : {len(df_raw):,}")

    # 1. Deduplication
    before = len(df_raw)
    df = df_raw.drop_duplicates()
    print(f"  Duplicates removed : {before - len(df)}")

    # 2. Missing values
    df["rating"]      = df["rating"].fillna(df["rating"].median())
    df["votes"]       = df["votes"].fillna(df["votes"].median()).astype(int)
    df["revenue_usd"] = df["revenue_usd"].fillna(0)
    print(f"  Missing values imputed (rating, votes, revenue)")

    # 3. Type casting
    df["year"] = df["year"].astype(int)

    # 4. Feature engineering
    df["primary_genre"] = df["genre"].str.split("|").str[0]
    df["profit_usd"]    = df["revenue_usd"] - df["budget_usd"]
    df["roi"]           = (df["profit_usd"] / df["budget_usd"].replace(0, np.nan)).round(3)
    df["decade"]        = (df["year"] // 10 * 10).astype(str) + "s"
    df["soup"]          = (df["title"].str.lower() + " " +
                           df["genre"].str.replace("|", " ").str.lower() + " " +
                           df["director"].str.lower())

    print(f"  Clean records      : {len(df):,}")
    return df.reset_index(drop=True)
