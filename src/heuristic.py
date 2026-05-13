"""
Heuristic algorithms for Graph Coloring problem.
Fast approximation algorithms that may not guarantee optimal solutions.
"""

import time
import random
from typing import Dict, List, Tuple
try:
    from .graph import Graph
except ImportError:
    from graph import Graph


class HeuristicColoring:
    """
    Heuristic algorithm implementations for graph coloring.

    Implements various greedy strategies for fast graph coloring.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the heuristic coloring algorithm.

        Args:
            graph: The graph to be colored
        """
        self.graph = graph
        self.vertices = graph.get_vertices()
        self.num_vertices = len(self.vertices)

    def greedy_coloring(self) -> Tuple[Dict[int, int], int, float]:
        """
        Basic greedy coloring algorithm.

        Colors vertices in order, assigning the smallest possible color
        that doesn't conflict with already colored neighbors.

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color
            - num_colors: Number of colors used
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        coloring = {}

        for vertex in self.vertices:
            # Get colors of already colored neighbors
            neighbor_colors = {
                coloring[neighbor]
                for neighbor in self.graph.get_neighbors(vertex)
                if neighbor in coloring
            }

            # Find the smallest color not used by neighbors
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color

        elapsed_time = time.time() - start_time
        num_colors = len(set(coloring.values()))
        return coloring, num_colors, elapsed_time

    def largest_degree_first(self) -> Tuple[Dict[int, int], int, float]:
        """
        Largest Degree First (Welsh-Powell) algorithm.

        Colors vertices in decreasing order of degree.

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color
            - num_colors: Number of colors used
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        # Sort vertices by degree in descending order
        sorted_vertices = sorted(
            self.vertices,
            key=lambda v: self.graph.degree(v),
            reverse=True
        )

        coloring = {}

        for vertex in sorted_vertices:
            # Get colors of already colored neighbors
            neighbor_colors = {
                coloring[neighbor]
                for neighbor in self.graph.get_neighbors(vertex)
                if neighbor in coloring
            }

            # Find the smallest color not used by neighbors
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color

        elapsed_time = time.time() - start_time
        num_colors = len(set(coloring.values()))
        return coloring, num_colors, elapsed_time

    def smallest_degree_last(self) -> Tuple[Dict[int, int], int, float]:
        """
        Smallest Degree Last algorithm.

        Removes vertex with smallest degree iteratively, then colors
        in reverse order of removal.

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color
            - num_colors: Number of colors used
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        # Create a copy of the graph for modification
        temp_graph = Graph()
        for v in self.vertices:
            temp_graph.add_vertex(v)
        for u, v in self.graph.get_edges():
            temp_graph.add_edge(u, v)

        removal_order = []
        remaining_vertices = set(self.vertices)

        # Remove vertices in smallest degree first order
        while remaining_vertices:
            # Find vertex with minimum degree
            min_vertex = min(
                remaining_vertices,
                key=lambda v: temp_graph.degree(v)
            )

            removal_order.append(min_vertex)
            remaining_vertices.remove(min_vertex)

            # Remove vertex and its edges from temp graph
            neighbors = temp_graph.get_neighbors(min_vertex)
            for neighbor in neighbors:
                if temp_graph.has_edge(min_vertex, neighbor):
                    temp_graph.graph.remove_edge(min_vertex, neighbor)
            temp_graph.graph.remove_node(min_vertex)

        # Color in reverse order of removal
        coloring = {}
        for vertex in reversed(removal_order):
            neighbor_colors = {
                coloring[neighbor]
                for neighbor in self.graph.get_neighbors(vertex)
                if neighbor in coloring
            }

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color

        elapsed_time = time.time() - start_time
        num_colors = len(set(coloring.values()))
        return coloring, num_colors, elapsed_time

    def dsatur_coloring(self) -> Tuple[Dict[int, int], int, float]:
        """
        DSATUR algorithm (Degree of Saturation).

        Colors vertex with highest saturation degree first.
        Saturation degree = number of different colors used by neighbors.

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color
            - num_colors: Number of colors used
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        coloring = {}
        uncolored = set(self.vertices)

        while uncolored:
            # Calculate saturation degree for each uncolored vertex
            sat_degrees = {}
            for vertex in uncolored:
                neighbor_colors = {
                    coloring[neighbor]
                    for neighbor in self.graph.get_neighbors(vertex)
                    if neighbor in coloring
                }
                sat_degrees[vertex] = len(neighbor_colors)

            # Choose vertex with highest saturation degree
            # Break ties by choosing vertex with highest degree
            max_sat = max(sat_degrees.values())
            candidates = [v for v, sat in sat_degrees.items() if sat == max_sat]

            if len(candidates) > 1:
                # Break ties by degree
                vertex = max(candidates, key=lambda v: self.graph.degree(v))
            else:
                vertex = candidates[0]

            # Color the chosen vertex
            neighbor_colors = {
                coloring[neighbor]
                for neighbor in self.graph.get_neighbors(vertex)
                if neighbor in coloring
            }

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color
            uncolored.remove(vertex)

        elapsed_time = time.time() - start_time
        num_colors = len(set(coloring.values()))
        return coloring, num_colors, elapsed_time

    def is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Check if a coloring is valid.

        Args:
            coloring: Dictionary mapping vertex to color

        Returns:
            bool: True if coloring is valid, False otherwise
        """
        for u, v in self.graph.get_edges():
            if coloring[u] == coloring[v]:
                return False
        return True

    def hybrid_coloring(self) -> Tuple[Dict[int, int], int, float]:
        """
        Hybrid algorithm that runs multiple heuristics and picks the best result.

        Returns:
            Tuple containing best coloring, number of colors, and total elapsed time
        """
        start_time = time.time()

        # Run all heuristic methods
        methods = ['dsatur', 'sdl', 'ldf', 'greedy']
        results = []

        for method in methods:
            if method == 'greedy':
                coloring, colors, _ = self.greedy_coloring()
            elif method == 'ldf':
                coloring, colors, _ = self.largest_degree_first()
            elif method == 'sdl':
                coloring, colors, _ = self.smallest_degree_last()
            elif method == 'dsatur':
                coloring, colors, _ = self.dsatur_coloring()

            results.append((coloring, colors, method))

        # Choose the result with minimum colors
        best_coloring, best_colors, best_method = min(results, key=lambda x: x[1])

        elapsed_time = time.time() - start_time
        return best_coloring, best_colors, elapsed_time

    def improved_greedy_coloring(self) -> Tuple[Dict[int, int], int, float]:
        """
        Improved greedy algorithm with better vertex ordering.
        Uses degree-based ordering with tie-breaking strategies.

        Returns:
            Tuple containing coloring, number of colors, and elapsed time
        """
        start_time = time.time()

        # Sort vertices by degree (descending), then by number of colored neighbors
        coloring = {}

        # Initial ordering by degree
        vertices_by_degree = sorted(
            self.vertices,
            key=lambda v: (self.graph.degree(v), random.random()),
            reverse=True
        )

        for vertex in vertices_by_degree:
            # Get colors of already colored neighbors
            neighbor_colors = {
                coloring[neighbor]
                for neighbor in self.graph.get_neighbors(vertex)
                if neighbor in coloring
            }

            # Find the smallest color not used by neighbors
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[vertex] = color

        elapsed_time = time.time() - start_time
        num_colors = len(set(coloring.values()))
        return coloring, num_colors, elapsed_time

    def solve(self, method: str = 'greedy') -> Tuple[Dict[int, int], int, float]:
        """
        Solve graph coloring using specified heuristic method.

        Args:
            method: Algorithm to use ('greedy', 'ldf', 'sdl', 'dsatur', 'hybrid', 'improved_greedy')

        Returns:
            Tuple containing coloring, number of colors, and elapsed time
        """
        if method == 'greedy':
            return self.greedy_coloring()
        elif method == 'ldf':
            return self.largest_degree_first()
        elif method == 'sdl':
            return self.smallest_degree_last()
        elif method == 'dsatur':
            return self.dsatur_coloring()
        elif method == 'hybrid':
            return self.hybrid_coloring()
        elif method == 'improved_greedy':
            return self.improved_greedy_coloring()
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(self) -> Dict[str, Dict[str, any]]:
        """
        Compare all heuristic methods on the same graph.

        Returns:
            Dictionary containing results for each method
        """
        methods = ['greedy', 'ldf', 'sdl', 'dsatur', 'hybrid', 'improved_greedy']
        results = {}

        for method in methods:
            coloring, num_colors, time_taken = self.solve(method)
            results[method] = {
                'coloring': coloring,
                'num_colors': num_colors,
                'time_taken': time_taken,
                'is_valid': self.is_valid_coloring(coloring)
            }

        return results