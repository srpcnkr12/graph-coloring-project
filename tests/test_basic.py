"""
Basic tests for the Graph Coloring project.
"""

import unittest
import sys
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import Graph, generate_complete_graph, generate_cycle_graph
from brute_force import BruteForceColoring
from heuristic import HeuristicColoring


class TestGraph(unittest.TestCase):
    """Test the Graph class."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        g = Graph()
        self.assertEqual(g.num_vertices, 0)
        self.assertEqual(g.num_edges, 0)

    def test_add_vertex(self):
        """Test adding vertices."""
        g = Graph()
        g.add_vertex(0)
        g.add_vertex(1)
        self.assertEqual(g.num_vertices, 2)
        self.assertIn(0, g.get_vertices())
        self.assertIn(1, g.get_vertices())

    def test_add_edge(self):
        """Test adding edges."""
        g = Graph()
        g.add_vertex(0)
        g.add_vertex(1)
        g.add_edge(0, 1)
        self.assertEqual(g.num_edges, 1)
        self.assertTrue(g.has_edge(0, 1))
        self.assertIn(1, g.get_neighbors(0))
        self.assertIn(0, g.get_neighbors(1))


class TestBruteForce(unittest.TestCase):
    """Test the Brute Force algorithm."""

    def test_triangle_graph(self):
        """Test coloring a triangle (K3)."""
        g = generate_complete_graph(3)
        bf = BruteForceColoring(g)
        coloring, num_colors, _ = bf.solve()

        self.assertTrue(bf.is_valid_coloring(coloring))
        self.assertEqual(num_colors, 3)  # Triangle needs 3 colors

    def test_path_graph(self):
        """Test coloring a path graph."""
        g = Graph()
        for i in range(4):
            g.add_vertex(i)
        # Create path: 0-1-2-3
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        bf = BruteForceColoring(g)
        coloring, num_colors, _ = bf.solve()

        self.assertTrue(bf.is_valid_coloring(coloring))
        self.assertEqual(num_colors, 2)  # Path needs 2 colors


class TestHeuristic(unittest.TestCase):
    """Test the Heuristic algorithms."""

    def test_greedy_triangle(self):
        """Test greedy algorithm on triangle."""
        g = generate_complete_graph(3)
        h = HeuristicColoring(g)
        coloring, num_colors, _ = h.greedy_coloring()

        self.assertTrue(h.is_valid_coloring(coloring))
        self.assertGreaterEqual(num_colors, 3)  # Should be exactly 3, might be more

    def test_all_methods_valid(self):
        """Test that all heuristic methods produce valid colorings."""
        g = generate_cycle_graph(5)
        h = HeuristicColoring(g)

        methods = ['greedy', 'ldf', 'sdl', 'dsatur']
        for method in methods:
            with self.subTest(method=method):
                coloring, _, _ = h.solve(method)
                self.assertTrue(h.is_valid_coloring(coloring))


class TestComplexity(unittest.TestCase):
    """Test complexity edge cases."""

    def test_single_vertex(self):
        """Test single vertex graph."""
        g = Graph()
        g.add_vertex(0)

        # Brute force
        bf = BruteForceColoring(g)
        bf_coloring, bf_colors, _ = bf.solve()
        self.assertEqual(bf_colors, 1)

        # Heuristic
        h = HeuristicColoring(g)
        h_coloring, h_colors, _ = h.greedy_coloring()
        self.assertEqual(h_colors, 1)

    def test_empty_graph(self):
        """Test empty graph."""
        g = Graph()

        # Brute force
        bf = BruteForceColoring(g)
        bf_coloring, bf_colors, _ = bf.solve()
        self.assertEqual(bf_colors, 0)

        # Heuristic
        h = HeuristicColoring(g)
        h_coloring, h_colors, _ = h.greedy_coloring()
        self.assertEqual(h_colors, 0)


if __name__ == '__main__':
    unittest.main()