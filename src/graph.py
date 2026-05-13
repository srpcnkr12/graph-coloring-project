"""
Graph representation and basic operations for the Graph Coloring problem.
"""

import networkx as nx
import random
from typing import Dict, List, Set, Optional


class Graph:
    """
    Graph class for representing undirected graphs for coloring problems.
    """

    def __init__(self):
        """Initialize an empty graph."""
        self.graph = nx.Graph()
        self.num_vertices = 0
        self.num_edges = 0

    def add_vertex(self, vertex: int) -> None:
        """Add a vertex to the graph."""
        self.graph.add_node(vertex)
        self.num_vertices = self.graph.number_of_nodes()

    def add_edge(self, u: int, v: int) -> None:
        """Add an edge between vertices u and v."""
        self.graph.add_edge(u, v)
        self.num_edges = self.graph.number_of_edges()

    def get_neighbors(self, vertex: int) -> List[int]:
        """Get all neighbors of a given vertex."""
        return list(self.graph.neighbors(vertex))

    def get_vertices(self) -> List[int]:
        """Get all vertices in the graph."""
        return list(self.graph.nodes())

    def get_edges(self) -> List[tuple]:
        """Get all edges in the graph."""
        return list(self.graph.edges())

    def degree(self, vertex: int) -> int:
        """Get the degree of a vertex."""
        return self.graph.degree(vertex)

    def has_edge(self, u: int, v: int) -> bool:
        """Check if there is an edge between u and v."""
        return self.graph.has_edge(u, v)

    def to_networkx(self) -> nx.Graph:
        """Return the NetworkX graph object."""
        return self.graph


def generate_random_graph(num_vertices: int, num_edges: int) -> Graph:
    """
    Generate a random graph with specified number of vertices and edges.

    Args:
        num_vertices: Number of vertices in the graph
        num_edges: Number of edges in the graph

    Returns:
        Graph: A randomly generated graph
    """
    g = Graph()

    # Add vertices
    for i in range(num_vertices):
        g.add_vertex(i)

    # Add random edges
    edges_added = 0
    max_edges = num_vertices * (num_vertices - 1) // 2

    if num_edges > max_edges:
        num_edges = max_edges

    while edges_added < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)

        if u != v and not g.has_edge(u, v):
            g.add_edge(u, v)
            edges_added += 1

    return g


def generate_complete_graph(num_vertices: int) -> Graph:
    """Generate a complete graph with n vertices."""
    g = Graph()
    for i in range(num_vertices):
        g.add_vertex(i)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            g.add_edge(i, j)

    return g


def generate_cycle_graph(num_vertices: int) -> Graph:
    """Generate a cycle graph with n vertices."""
    g = Graph()
    for i in range(num_vertices):
        g.add_vertex(i)

    for i in range(num_vertices):
        g.add_edge(i, (i + 1) % num_vertices)

    return g