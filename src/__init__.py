"""
Graph Coloring Project - CS 301

This package implements and compares different algorithms for the Graph Coloring problem.
"""

from .graph import Graph, generate_random_graph, generate_complete_graph, generate_cycle_graph
from .brute_force import BruteForceColoring
from .heuristic import HeuristicColoring
from .visualization import GraphVisualizer

__version__ = "1.0.0"
__all__ = [
    'Graph',
    'generate_random_graph',
    'generate_complete_graph',
    'generate_cycle_graph',
    'BruteForceColoring',
    'HeuristicColoring',
    'GraphVisualizer'
]