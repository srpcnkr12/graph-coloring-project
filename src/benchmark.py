"""
Comprehensive benchmark suite for analyzing heuristic algorithm performance.
"""

import time
import random
import pandas as pd
from typing import Dict, List, Tuple
from graph import Graph, generate_random_graph, generate_complete_graph, generate_cycle_graph
from brute_force import BruteForceColoring
from heuristic import HeuristicColoring


class BenchmarkSuite:
    """
    Comprehensive benchmarking tools for graph coloring algorithms.
    """

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = []

    def run_accuracy_analysis(self, max_vertices: int = 9,
                            samples_per_size: int = 10) -> pd.DataFrame:
        """
        Run comprehensive accuracy analysis comparing heuristics to optimal solutions.

        Args:
            max_vertices: Maximum graph size to test (brute force limit)
            samples_per_size: Number of random graphs per vertex count

        Returns:
            DataFrame with detailed results
        """
        print("Running accuracy analysis...")
        results = []

        for n_vertices in range(3, max_vertices + 1):
            print(f"Testing graphs with {n_vertices} vertices...")

            for sample in range(samples_per_size):
                # Generate different types of graphs
                graphs = self._generate_test_graphs(n_vertices)

                for graph_type, graph in graphs.items():
                    result = self._analyze_single_graph(graph, graph_type, n_vertices, sample)
                    results.append(result)

        df = pd.DataFrame(results)
        self.results = df
        return df

    def _generate_test_graphs(self, n_vertices: int) -> Dict[str, Graph]:
        """Generate different types of test graphs."""
        graphs = {}

        # Random sparse graph
        max_edges = n_vertices * (n_vertices - 1) // 2
        sparse_edges = min(n_vertices, max_edges // 3)
        graphs['sparse_random'] = generate_random_graph(n_vertices, sparse_edges)

        # Random dense graph
        dense_edges = min(max_edges // 2, max_edges)
        graphs['dense_random'] = generate_random_graph(n_vertices, dense_edges)

        # Cycle graph
        if n_vertices >= 3:
            graphs['cycle'] = generate_cycle_graph(n_vertices)

        # Complete graph (only for small sizes due to exponential coloring)
        if n_vertices <= 6:
            graphs['complete'] = generate_complete_graph(n_vertices)

        # Path graph
        path_graph = Graph()
        for i in range(n_vertices):
            path_graph.add_vertex(i)
        for i in range(n_vertices - 1):
            path_graph.add_edge(i, i + 1)
        graphs['path'] = path_graph

        return graphs

    def _analyze_single_graph(self, graph: Graph, graph_type: str,
                            n_vertices: int, sample_id: int) -> Dict:
        """Analyze a single graph with all algorithms."""

        # Get optimal solution with brute force
        bf = BruteForceColoring(graph)
        optimal_coloring, optimal_colors, bf_time = bf.solve()

        # Test all heuristic methods
        heuristic = HeuristicColoring(graph)
        methods = ['dsatur', 'ldf', 'hybrid']

        result = {
            'vertices': n_vertices,
            'edges': graph.num_edges,
            'graph_type': graph_type,
            'sample_id': sample_id,
            'optimal_colors': optimal_colors,
            'bf_time': bf_time
        }

        for method in methods:
            h_coloring, h_colors, h_time = heuristic.solve(method)
            error_rate = (h_colors - optimal_colors) / optimal_colors if optimal_colors > 0 else 0

            result[f'{method}_colors'] = h_colors
            result[f'{method}_time'] = h_time
            result[f'{method}_error_rate'] = error_rate
            result[f'{method}_valid'] = heuristic.is_valid_coloring(h_coloring)

        return result

    def analyze_accuracy_by_graph_type(self) -> pd.DataFrame:
        """Analyze accuracy breakdown by graph type."""
        if self.results is None or len(self.results) == 0:
            print("No results available. Run accuracy analysis first.")
            return pd.DataFrame()

        methods = ['dsatur', 'ldf', 'hybrid']
        summary = []

        for graph_type in self.results['graph_type'].unique():
            type_data = self.results[self.results['graph_type'] == graph_type]

            for method in methods:
                error_col = f'{method}_error_rate'
                if error_col in type_data.columns:
                    mean_error = type_data[error_col].mean()
                    std_error = type_data[error_col].std()
                    max_error = type_data[error_col].max()
                    failure_rate = (type_data[error_col] > 0).mean()

                    summary.append({
                        'graph_type': graph_type,
                        'method': method,
                        'mean_error_rate': mean_error,
                        'std_error_rate': std_error,
                        'max_error_rate': max_error,
                        'failure_rate': failure_rate,
                        'sample_count': len(type_data)
                    })

        return pd.DataFrame(summary)

    def find_worst_cases(self, method: str = 'greedy', top_n: int = 10) -> pd.DataFrame:
        """Find worst performing cases for a specific method."""
        if self.results is None or len(self.results) == 0:
            print("No results available. Run accuracy analysis first.")
            return pd.DataFrame()

        error_col = f'{method}_error_rate'
        if error_col not in self.results.columns:
            print(f"Method {method} not found in results.")
            return pd.DataFrame()

        worst_cases = self.results.nlargest(top_n, error_col)
        return worst_cases[['vertices', 'edges', 'graph_type', 'optimal_colors',
                          f'{method}_colors', error_col]]

    def compare_methods_overall(self) -> Dict[str, Dict[str, float]]:
        """Compare all methods with overall statistics."""
        if self.results is None or len(self.results) == 0:
            print("No results available. Run accuracy analysis first.")
            return {}

        methods = ['dsatur', 'ldf', 'hybrid']
        comparison = {}

        for method in methods:
            error_col = f'{method}_error_rate'
            time_col = f'{method}_time'

            if error_col in self.results.columns:
                comparison[method] = {
                    'mean_error_rate': self.results[error_col].mean(),
                    'median_error_rate': self.results[error_col].median(),
                    'std_error_rate': self.results[error_col].std(),
                    'max_error_rate': self.results[error_col].max(),
                    'failure_rate': (self.results[error_col] > 0).mean(),
                    'mean_time': self.results[time_col].mean(),
                    'perfect_solutions': (self.results[error_col] == 0).sum(),
                    'total_tests': len(self.results)
                }

        return comparison

    def save_results(self, filename: str = "results/benchmark_results.csv"):
        """Save benchmark results to CSV."""
        if self.results is not None and len(self.results) > 0:
            self.results.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No results to save.")

    def generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for improving heuristic performance."""
        if self.results is None or len(self.results) == 0:
            return ["Run benchmark analysis first to generate recommendations."]

        recommendations = []
        comparison = self.compare_methods_overall()

        # Find best performing method
        best_method = min(comparison.keys(),
                         key=lambda m: comparison[m]['mean_error_rate'])
        worst_method = max(comparison.keys(),
                          key=lambda m: comparison[m]['mean_error_rate'])

        recommendations.append(f"Best performing method: {best_method.upper()} "
                             f"(avg error: {comparison[best_method]['mean_error_rate']:.3f})")

        recommendations.append(f"Worst performing method: {worst_method.upper()} "
                             f"(avg error: {comparison[worst_method]['mean_error_rate']:.3f})")

        # Analyze by graph type
        type_analysis = self.analyze_accuracy_by_graph_type()
        if not type_analysis.empty:
            worst_graph_type = type_analysis.loc[
                type_analysis['mean_error_rate'].idxmax(), 'graph_type']
            recommendations.append(f"Most challenging graph type: {worst_graph_type}")

        # General recommendations
        overall_error = sum(comparison[m]['mean_error_rate'] for m in comparison) / len(comparison)
        if overall_error > 0.1:
            recommendations.append("HIGH PRIORITY: Overall error rate > 10%. Consider:")
            recommendations.append("- Implementing meta-heuristic algorithms")
            recommendations.append("- Adding local search improvement phases")
            recommendations.append("- Using hybrid approaches")

        return recommendations