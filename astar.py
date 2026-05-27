"""
A* pathfinding applied to a genre-relationship graph.
Used to navigate from one genre cluster to another with minimum cost.
"""

import heapq
from typing import Dict, List, Tuple, Optional


def _heuristic(node: str, goal: str) -> int:
    """Simple heuristic: 0 (Dijkstra-style) since we have no coordinates."""
    return 0


def astar_path(
    graph: Dict[str, Dict[str, int]],
    start: str,
    goal: str,
) -> Tuple[List[str], int]:
    """
    A* search on a weighted directed genre graph.

    Parameters
    ----------
    graph : adjacency dict  {node: {neighbor: cost, ...}, ...}
    start : source node
    goal  : target node

    Returns
    -------
    (path, total_cost)
    """
    # priority queue entries: (f_score, g_score, node, path)
    open_heap: list = []
    heapq.heappush(open_heap, (0 + _heuristic(start, goal), 0, start, [start]))
    visited: set = set()

    while open_heap:
        f, g, current, path = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path, g

        for neighbor, cost in graph.get(current, {}).items():
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (new_f, new_g, neighbor, path + [neighbor]))

    return [], -1  # no path found


def build_genre_graph(df) -> Dict[str, Dict[str, int]]:
    """
    Build a genre co-occurrence graph from the dataset.
    Edge weight = 10 - floor(avg_rating_of_shared_movies).
    """
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    graph: Dict[str, Dict[str, int]] = defaultdict(dict)
    df["genres_list"] = df["genre"].str.split("|")

    for _, row in df.iterrows():
        genres = row["genres_list"]
        rating = row["rating"]
        if len(genres) >= 2:
            g1, g2 = genres[0], genres[1]
            cost = max(1, int(10 - rating))
            if g2 not in graph[g1] or graph[g1][g2] > cost:
                graph[g1][g2] = cost

    return dict(graph)
