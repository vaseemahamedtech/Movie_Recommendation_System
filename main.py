"""
Data-Driven Movie Recommendation System
Author: Vaseem Ahamed
GitHub: https://github.com/vaseemahamedtech
"""

import sys

# Reconfigure stdout/stderr to utf-8 for emoji support on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from data_loader import load_and_clean_data
from eda import run_eda
from recommender import build_recommender, get_recommendations
from astar import astar_path
from visualise import build_folium_map
from theater_recommender import run_theater_recommender



def run_movie_pipeline():
    print("\n🎬 Starting Content-Based Movie Recommendation Pipeline...")
    df = load_and_clean_data("data/movies.csv")
    print(f"\n✅  Loaded {len(df):,} clean movie records.\n")

    run_eda(df)

    model, tfidf_matrix = build_recommender(df)

    test_movies = ["The Dark Knight", "Inception", "Interstellar"]
    for title in test_movies:
        recs = get_recommendations(title, df, model, tfidf_matrix, top_n=5)
        print(f"\n🎬  Top 5 recommendations for '{title}':")
        for i, (movie, score) in enumerate(recs, 1):
            print(f"  {i}. {movie:<40} (similarity: {score:.4f})")

    graph = {
        "Action":    {"Adventure": 2, "Sci-Fi": 3},
        "Adventure": {"Fantasy": 1, "Family": 4},
        "Sci-Fi":    {"Thriller": 2, "Mystery": 5},
        "Fantasy":   {"Animation": 3},
        "Thriller":  {"Crime": 1, "Drama": 2},
        "Crime":     {"Drama": 1},
        "Drama":     {},
        "Mystery":   {"Drama": 2},
        "Family":    {"Animation": 1},
        "Animation": {},
    }
    path, cost = astar_path(graph, "Action", "Animation")
    print(f"\n🔍  A* Genre Path: {' → '.join(path)}  (cost: {cost})")

    build_folium_map(df, output_file="output/movie_map.html")
    print("\n🗺  Geospatial map saved to output/movie_map.html")
    print("\n✅ Movie recommendation pipeline complete.")


def main():
    while True:
        print("\n" + "=" * 60)
        print("  🎥  Movie Recommendation System — Vaseem Ahamed  🎥")
        print("=" * 60)
        print("1. Run Content-Based Movie & Genre Recommendation Pipeline")
        print("2. Run Location-Based Theater Recommendation (A* Heuristic)")
        print("3. Run Both Pipelines")
        print("4. Exit")
        print("=" * 60)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "1":
            run_movie_pipeline()
        elif choice == "2":
            run_theater_recommender()
        elif choice == "3":
            run_movie_pipeline()
            run_theater_recommender()
        elif choice == "4":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()

