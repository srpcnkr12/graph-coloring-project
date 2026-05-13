"""
Main module for the Graph Coloring project.
Provides command-line interface and demonstration of algorithms.
"""

import argparse
import sys
import time
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from graph import Graph, generate_random_graph, generate_complete_graph, generate_cycle_graph
from brute_force import BruteForceColoring
from heuristic import HeuristicColoring
from visualization import GraphVisualizer


class GraphColoringProject:
    """
    Main class for the Graph Coloring project.
    """

    def __init__(self):
        """Initialize the project."""
        self.visualizer = GraphVisualizer()

    def run_demo(self):
        """Run a demonstration of the graph coloring algorithms."""
        print("=" * 60)
        print("GRAPH COLORING PROJECT - CS 301")
        print("=" * 60)

        # Create different types of graphs for demonstration
        graphs = {
            "Small Random": generate_random_graph(6, 8),
            "Complete K4": generate_complete_graph(4),
            "Cycle C5": generate_cycle_graph(5),
            "Medium Random": generate_random_graph(8, 12)
        }

        for graph_name, graph in graphs.items():
            print(f"\n{'-' * 40}")
            print(f"Testing: {graph_name}")
            print(f"Vertices: {graph.num_vertices}, Edges: {graph.num_edges}")
            print(f"{'-' * 40}")

            self.analyze_graph(graph, graph_name)

    def analyze_graph(self, graph: Graph, graph_name: str):
        """
        Analyze a graph using both brute force and heuristic algorithms.

        Args:
            graph: Graph to analyze
            graph_name: Name of the graph for display
        """
        results = {}

        # Brute Force Analysis
        print("\n1. BRUTE FORCE ALGORITHM:")
        if graph.num_vertices <= 8:  # Only run for small graphs
            bf = BruteForceColoring(graph)
            bf_coloring, bf_colors, bf_time = bf.solve()
            results['Brute Force'] = {
                'coloring': bf_coloring,
                'num_colors': bf_colors,
                'time_taken': bf_time,
                'is_optimal': True
            }
            print(f"   Optimal coloring: {bf_coloring}")
            print(f"   Chromatic number: {bf_colors}")
            print(f"   Time taken: {bf_time:.6f} seconds")
        else:
            print("   Skipped (graph too large for brute force)")

        # Heuristic Analysis
        print("\n2. HEURISTIC ALGORITHMS:")
        heuristic = HeuristicColoring(graph)
        heuristic_results = heuristic.compare_methods()

        for method, result in heuristic_results.items():
            method_name = method.upper()
            results[method_name] = result
            print(f"\n   {method_name}:")
            print(f"     Coloring: {result['coloring']}")
            print(f"     Colors used: {result['num_colors']}")
            print(f"     Time taken: {result['time_taken']:.6f} seconds")
            print(f"     Valid: {result['is_valid']}")

        # Compare with optimal if available
        if 'Brute Force' in results:
            optimal_colors = results['Brute Force']['num_colors']
            print(f"\n3. QUALITY ANALYSIS:")
            for method, result in results.items():
                if method != 'Brute Force':
                    ratio = result['num_colors'] / optimal_colors
                    print(f"   {method}: {result['num_colors']}/{optimal_colors} = {ratio:.2f}x optimal")

        # Visualize the best result
        best_method = min(results.keys(), key=lambda k: results[k]['num_colors'])
        best_coloring = results[best_method]['coloring']

        print(f"\n4. VISUALIZATION:")
        print(f"   Plotting best result ({best_method})...")

        try:
            self.visualizer.plot_graph_coloring(
                graph, best_coloring,
                title=f"{graph_name} - {best_method}",
                save_path=f"results/{graph_name.lower().replace(' ', '_')}_coloring.png"
            )
        except Exception as e:
            print(f"   Visualization failed: {e}")

    def run_performance_test(self, max_vertices: int = 10):
        """
        Run performance tests on graphs of increasing size.

        Args:
            max_vertices: Maximum number of vertices to test
        """
        print(f"\n{'=' * 60}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'=' * 60}")

        sizes = list(range(3, max_vertices + 1))
        bf_times = []
        heuristic_times = []

        for size in sizes:
            print(f"\nTesting graph with {size} vertices...")
            graph = generate_random_graph(size, min(size * (size - 1) // 4, 20))

            # Brute force (only for small graphs)
            if size <= 8:
                bf = BruteForceColoring(graph)
                _, _, bf_time = bf.solve()
                bf_times.append(bf_time)
                print(f"  Brute Force: {bf_time:.6f}s")
            else:
                bf_times.append(float('inf'))
                print(f"  Brute Force: Skipped (too large)")

            # Heuristic
            heuristic = HeuristicColoring(graph)
            _, _, h_time = heuristic.greedy_coloring()
            heuristic_times.append(h_time)
            print(f"  Heuristic: {h_time:.6f}s")

        # Plot complexity analysis
        valid_bf_times = [t for t in bf_times if t != float('inf')]
        valid_sizes = sizes[:len(valid_bf_times)]

        if valid_bf_times:
            self.visualizer.plot_complexity_analysis(
                valid_sizes, valid_bf_times, heuristic_times[:len(valid_bf_times)],
                save_path="results/complexity_analysis.png"
            )

    def interactive_mode(self):
        """Run interactive mode for custom graph creation."""
        print(f"\n{'=' * 60}")
        print("INTERACTIVE MODE")
        print(f"{'=' * 60}")

        while True:
            print("\nOptions:")
            print("1. Create random graph")
            print("2. Create complete graph")
            print("3. Create cycle graph")
            print("4. Exit")

            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '4':
                break

            try:
                if choice == '1':
                    vertices = int(input("Number of vertices: "))
                    edges = int(input("Number of edges: "))
                    graph = generate_random_graph(vertices, edges)
                    name = "Custom Random"

                elif choice == '2':
                    vertices = int(input("Number of vertices: "))
                    graph = generate_complete_graph(vertices)
                    name = f"Complete K{vertices}"

                elif choice == '3':
                    vertices = int(input("Number of vertices: "))
                    graph = generate_cycle_graph(vertices)
                    name = f"Cycle C{vertices}"

                else:
                    print("Invalid choice!")
                    continue

                print(f"\nAnalyzing {name}...")
                self.analyze_graph(graph, name)

            except ValueError:
                print("Invalid input! Please enter numbers only.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Graph Coloring Project')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration on sample graphs')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance analysis')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive mode')
    parser.add_argument('--max-vertices', type=int, default=10,
                       help='Maximum vertices for performance test')

    args = parser.parse_args()

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    project = GraphColoringProject()

    if args.demo:
        project.run_demo()
    elif args.performance:
        project.run_performance_test(args.max_vertices)
    elif args.interactive:
        project.interactive_mode()
    else:
        # Default: run demo
        project.run_demo()


if __name__ == "__main__":
    main()