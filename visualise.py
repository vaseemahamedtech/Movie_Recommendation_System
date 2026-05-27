"""
Geospatial visualisation with Folium.
Creates an interactive HTML map showing movie production origins,
rated by colour intensity.
"""

import os
import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np


def _rating_to_color(rating: float) -> str:
    """Map a 1-10 rating to a hex colour (red → yellow → green)."""
    r = min(max(rating, 1), 10)
    if r >= 7.5:
        return "#2ECC71"   # green
    elif r >= 5.5:
        return "#F1C40F"   # yellow
    else:
        return "#E74C3C"   # red


def build_folium_map(df: pd.DataFrame, output_file: str = "output/movie_map.html"):
    """
    Generate an interactive Folium map:
    - Clustered circle markers, colour-coded by rating
    - Heat-map layer for production density
    - Mini-map control & layer toggles
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Sample for performance (max 500 points on the map)
    sample = df.sample(min(500, len(df)), random_state=42)

    centre = [20.0, 10.0]
    m = folium.Map(location=centre, zoom_start=3, tiles="CartoDB dark_matter")

    # ── Marker cluster layer ──────────────────────────────────────────────────
    cluster = MarkerCluster(name="Movies (clustered)").add_to(m)
    for _, row in sample.iterrows():
        color = _rating_to_color(row["rating"])
        popup_html = f"""
        <b>{row['title']} ({row['year']})</b><br>
        Genre: {row['genre']}<br>
        Language: {row['language']}<br>
        Rating: ⭐ {row['rating']:.1f}<br>
        Votes: {int(row['votes']):,}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['title']} ⭐{row['rating']:.1f}",
        ).add_to(cluster)

    # ── Heat-map layer ────────────────────────────────────────────────────────
    heat_data = [[r["lat"], r["lon"], r["rating"]/10]
                 for _, r in sample.iterrows()]
    HeatMap(heat_data, name="Production Density", radius=18,
            gradient={"0.3": "blue", "0.65": "lime", "1": "red"}).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;
                background:rgba(0,0,0,0.75);padding:12px 18px;
                border-radius:8px;color:white;font-family:sans-serif;font-size:13px;">
      <b>⭐ Rating Legend</b><br>
      <span style="color:#2ECC71">●</span> ≥ 7.5 (High)<br>
      <span style="color:#F1C40F">●</span> 5.5–7.4 (Mid)<br>
      <span style="color:#E74C3C">●</span> &lt; 5.5 (Low)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    folium.plugins.MiniMap(toggle_display=True).add_to(m)

    m.save(output_file)
    return output_file
