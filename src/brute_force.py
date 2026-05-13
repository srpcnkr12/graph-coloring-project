"""
Brute Force algorithm for Graph Coloring problem.
Guarantees optimal solution but has exponential time complexity.
"""

import itertools
import time
from typing import Dict, List, Optional, Tuple
try:
    from .graph import Graph
except ImportError:
    from graph import Graph


class BruteForceColoring:
    """
    Brute Force algorithm implementation for graph coloring.

    This algorithm tries all possible color combinations to find
    the minimum number of colors needed for a valid coloring.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the brute force coloring algorithm.

        Args:
            graph: The graph to be colored
        """
        self.graph = graph
        self.vertices = graph.get_vertices()
        self.num_vertices = len(self.vertices)

    def is_valid_coloring(self, coloring: Dict[int, int]) -> bool:
        """
        Check if a coloring is valid (no adjacent vertices have same color).

        Args:
            coloring: Dictionary mapping vertex to color

        Returns:
            bool: True if coloring is valid, False otherwise
        """
        for u, v in self.graph.get_edges():
            if coloring[u] == coloring[v]:
                return False
        return True

    def solve(self) -> Tuple[Dict[int, int], int, float]:
        """
        Find the optimal coloring using brute force approach.

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color
            - min_colors: Minimum number of colors used
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        # Try from 1 color up to number of vertices
        for num_colors in range(1, self.num_vertices + 1):
            # Generate all possible color combinations
            for colors in itertools.product(range(num_colors), repeat=self.num_vertices):
                coloring = dict(zip(self.vertices, colors))

                if self.is_valid_coloring(coloring):
                    elapsed_time = time.time() - start_time
                    actual_colors_used = len(set(colors))
                    return coloring, actual_colors_used, elapsed_time

        # Fallback (should never reach here for valid graphs)
        elapsed_time = time.time() - start_time
        trivial_coloring = {v: i for i, v in enumerate(self.vertices)}
        return trivial_coloring, self.num_vertices, elapsed_time

    def solve_with_k_colors(self, k: int) -> Tuple[Optional[Dict[int, int]], bool, float]:
        """
        Check if the graph can be colored with exactly k colors.

        Args:
            k: Number of colors to use

        Returns:
            Tuple containing:
            - coloring: Dictionary mapping vertex to color (None if impossible)
            - is_possible: True if k-coloring exists, False otherwise
            - elapsed_time: Time taken to solve
        """
        start_time = time.time()

        # Generate all possible k-colorings
        for colors in itertools.product(range(k), repeat=self.num_vertices):
            coloring = dict(zip(self.vertices, colors))

            if self.is_valid_coloring(coloring):
                elapsed_time = time.time() - start_time
                return coloring, True, elapsed_time

        elapsed_time = time.time() - start_time
        return None, False, elapsed_time

    def get_chromatic_number(self) -> Tuple[int, float]:
        """
        Find the chromatic number of the graph.

        Returns:
            Tuple containing:
            - chromatic_number: Minimum number of colors needed
            - elapsed_time: Time taken to solve
        """
        _, min_colors, elapsed_time = self.solve()
        return min_colors, elapsed_time

    def analyze_complexity(self) -> Dict[str, any]:
        """
        Analyze the complexity for this specific graph instance.

        Returns:
            Dictionary containing complexity analysis
        """
        n = self.num_vertices
        worst_case_combinations = sum(k**n for k in range(1, n + 1))

        return {
            'vertices': n,
            'edges': self.graph.num_edges,
            'theoretical_time_complexity': f'O(k^n) where n={n}',
            'theoretical_space_complexity': f'O(n) where n={n}',
            'worst_case_combinations': worst_case_combinations
        }