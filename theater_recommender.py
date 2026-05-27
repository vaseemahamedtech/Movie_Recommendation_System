"""
Theater and Movie Recommendation using A* Heuristic Search.
Calculates the best theater option based on user location, ticket price, movie rating, and genre.
"""

import os
import sys
import heapq
import time
import urllib3
import pandas as pd
import matplotlib.pyplot as plt

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
except ImportError:
    Nominatim = None

    def geodesic(a, b):
        import math

        lat1, lon1 = map(math.radians, a)
        lat2, lon2 = map(math.radians, b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a_val = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a_val), math.sqrt(1 - a_val))
        distance_km = 6371.0 * c

        class Distance:
            def __init__(self, km):
                self.kilometers = km

        return Distance(distance_km)


def get_theater_data() -> pd.DataFrame:
    """Return the default theater movie dataset."""
    data = {
        'Theater': ['PVR Cinemas', 'INOX', 'Sathyam', 'AGS', 'Escape Cinemas'],
        'Movie': ['Fighter', '3 Idiots', 'Jawan', 'Love Today', 'The Conjuring'],
        'Genre': ['Action', 'Comedy', 'Action', 'Romance', 'Horror'],
        'Latitude': [13.0827, 13.0612, 13.0500, 13.0901, 13.0582],
        'Longitude': [80.2707, 80.2425, 80.2590, 80.2407, 80.2348],
        'TicketPrice': [220, 180, 150, 170, 200],
        'Rating': [4.5, 4.0, 4.8, 3.9, 4.2]
    }
    return pd.DataFrame(data)


def geocode_with_retry(place_name: str, max_retries: int = 3, delay: float = 1.0):
    """Geocode a place name into lat/lon with retry and exponential backoff."""
    if Nominatim is None:
        return None

    geolocator = Nominatim(user_agent="movie-recommender", timeout=3)
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(place_name)
            if location:
                return location
        except (urllib3.exceptions.ReadTimeoutError, urllib3.exceptions.MaxRetryError) as e:
            print(f"⚠️ Geocoding failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"⚠️ An unexpected geocoding error occurred: {e}")
            return None
    print("❌ Max retries reached. Geocoding failed.")
    return None


def heuristic(theater, user_location, preferred_genre) -> float:
    """
    Heuristic cost function for A* recommendation.
    Combines:
    - Distance (km)
    - Ticket price score (TicketPrice / 100)
    - Rating score (5 - Rating)
    - Genre penalty (+15 if not matching preferred genre)
    """
    theater_location = (theater['Latitude'], theater['Longitude'])
    distance = geodesic(user_location, theater_location).kilometers
    genre_penalty = 0 if theater['Genre'].lower() == preferred_genre.lower() else 15
    price_score = theater['TicketPrice'] / 100
    rating_score = 5 - theater['Rating']
    return distance + price_score + rating_score + genre_penalty


def a_star_recommendation(df: pd.DataFrame, user_location, preferred_genre):
    """
    A* Recommendation based on heuristic costs.
    Computes cost for each option, logs all costs, and returns the best recommendation.
    """
    open_list = []
    costs = []

    for idx, theater in df.iterrows():
        cost = heuristic(theater, user_location, preferred_genre)
        costs.append(cost)
        heapq.heappush(open_list, (cost, theater.to_dict()))

    # Add A* cost as a column
    df['AStarCost'] = costs

    print("\n📊 --- A* Cost Table ---")
    sorted_df = df[['Theater', 'Movie', 'Genre', 'TicketPrice', 'Rating', 'AStarCost']].sort_values(by='AStarCost')
    print(sorted_df.to_string(index=False))

    best_cost, best_theater = heapq.heappop(open_list)
    return best_theater


def plot_theaters(df: pd.DataFrame, user_location, recommended, output_path: str = "output/theater_map.png"):
    """Plot theater locations and recommendation on a 2D scatter plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Plot all theaters
    for _, row in df.iterrows():
        plt.scatter(row['Longitude'], row['Latitude'], c='blue', s=80, alpha=0.6)
        plt.text(row['Longitude'] + 0.0005, row['Latitude'],
                 f"{row['Theater']}\n({row['Movie']})", fontsize=9, fontweight='semibold')

    # Plot User location
    plt.scatter(user_location[1], user_location[0], c='green', marker='X', s=150, label='You (Current Location)')
    plt.text(user_location[1] + 0.0005, user_location[0], 'You', fontsize=10, color='green', fontweight='bold')

    # Plot Recommended theater
    plt.scatter(recommended['Longitude'], recommended['Latitude'], c='red', marker='*', s=250, label='Recommended')
    
    plt.title("Movie Theater Locations and Recommendation", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"💾 Saved theater map plot to {output_path}")
    if "--no-show" not in sys.argv:
        plt.show()
    else:
        plt.close()


def plot_a_star_costs(df: pd.DataFrame, user_location, preferred_genre, recommended, output_path: str = "output/astar_costs.png"):
    """Plot A* Heuristic Costs for each theater option."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    costs = []
    labels = []

    for _, theater in df.iterrows():
        cost = heuristic(theater, user_location, preferred_genre)
        costs.append(cost)
        labels.append(f"{theater['Theater']}\n({theater['Movie']})")

    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels, costs, color='skyblue', edgecolor='gray', height=0.6)
    
    # Highlight the recommended one in red/salmon
    for i, label in enumerate(labels):
        if recommended['Theater'] in label:
            bars[i].set_color('salmon')
            bars[i].set_edgecolor('red')

    plt.xlabel("Total A* Heuristic Cost (Lower is Better)", fontsize=12)
    plt.title("A* Cost for Each Theater Option", fontsize=14, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()  # top-down list
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"💾 Saved A* cost plot to {output_path}")
    if "--no-show" not in sys.argv:
        plt.show()
    else:
        plt.close()


def run_theater_recommender(place_name: str = None, preferred_genre: str = None):
    """Run the complete theater recommendation workflow."""
    print("\n" + "=" * 50)
    print("📍  A* Location-Based Theater Recommendation")
    print("=" * 50)

    df = get_theater_data()

    if not place_name:
        place_name = input("Enter your current location (e.g., Tambaram, Chennai Central): ").strip()
    if not preferred_genre:
        preferred_genre = input("Enter your preferred movie genre (e.g., Action, Comedy): ").strip()

    if not place_name or not preferred_genre:
        print("❌ Location and preferred genre cannot be empty. Returning to menu.")
        return

    # Get user location coordinates
    user_location = None
    if Nominatim is not None:
        print(f"🔍 Geocoding '{place_name}'...")
        location = geocode_with_retry(place_name)
        if location:
            user_location = (location.latitude, location.longitude)
            print(f"✅ Geocoding successful: {location.address} ({user_location[0]:.4f}, {user_location[1]:.4f})")
        else:
            print("⚠️ Geocoding failed. Falling back to coordinate input.")

    if user_location is None:
        # Fallback coordinate input
        print("\n📍 Coordinates needed (fallback):")
        try:
            lat = float(input("Enter your latitude (e.g., 13.0827): "))
            lon = float(input("Enter your longitude (e.g., 80.2707): "))
            user_location = (lat, lon)
        except ValueError:
            print("❌ Invalid coordinates. Returning to menu.")
            return

    # Run recommendation
    recommended = a_star_recommendation(df, user_location, preferred_genre)

    # Output results
    print("\n✨ --- Best Recommendation --- ✨")
    print(f"🎭 Theater       : {recommended['Theater']}")
    print(f"🎬 Movie         : {recommended['Movie']}")
    print(f"🏷️ Genre         : {recommended['Genre']}")
    distance_val = geodesic(user_location, (recommended['Latitude'], recommended['Longitude'])).km
    print(f"🚗 Distance      : {distance_val:.2f} km")
    print(f"💰 Ticket Price  : ₹{recommended['TicketPrice']}")
    print(f"⭐ Rating        : {recommended['Rating']}")
    print("=" * 50)

    # Generate plots
    print("\n🎨 Generating visualization plots...")
    plot_theaters(df, user_location, recommended)
    plot_a_star_costs(df, user_location, preferred_genre, recommended)
    print("✅ Visualization complete.")
