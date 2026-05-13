"""
Quality analysis module for heuristic algorithms.
Analyzes approximation ratios and solution quality compared to optimal solutions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path

try:
    from .graph import Graph, generate_random_graph, generate_complete_graph, generate_cycle_graph
    from .brute_force import BruteForceColoring
    from .heuristic import HeuristicColoring
except ImportError:
    from graph import Graph, generate_random_graph, generate_complete_graph, generate_cycle_graph
    from brute_force import BruteForceColoring
    from heuristic import HeuristicColoring


class QualityAnalyzer:
    """
    Analyzes solution quality of heuristic algorithms compared to optimal solutions.
    Calculates approximation ratios and quality metrics.
    """

    def __init__(self):
        """Initialize quality analyzer."""
        self.results = []

    def calculate_approximation_ratio(self, heuristic_colors: int, optimal_colors: int) -> float:
        """
        Calculate approximation ratio for a heuristic solution.

        Args:
            heuristic_colors: Number of colors used by heuristic
            optimal_colors: Optimal number of colors

        Returns:
            Approximation ratio (heuristic/optimal)
        """
        if optimal_colors == 0:
            return float('inf')
        return heuristic_colors / optimal_colors

    def analyze_single_graph(self, graph: Graph, graph_info: Dict) -> Dict:
        """
        Analyze quality of all heuristic methods on a single graph.

        Args:
            graph: Graph instance
            graph_info: Dictionary with graph metadata

        Returns:
            Dictionary with quality analysis results
        """
        # Get optimal solution
        bf = BruteForceColoring(graph)
        optimal_coloring, optimal_colors, bf_time = bf.solve()

        # Test all heuristic methods
        heuristic = HeuristicColoring(graph)
        methods = ['dsatur', 'ldf', 'hybrid']

        result = {
            'vertices': graph_info.get('vertices', graph.num_vertices),
            'edges': graph.num_edges,
            'graph_type': graph_info.get('type', 'unknown'),
            'optimal_colors': optimal_colors,
            'optimal_time': bf_time
        }

        for method in methods:
            h_coloring, h_colors, h_time = heuristic.solve(method)
            approx_ratio = self.calculate_approximation_ratio(h_colors, optimal_colors)
            error_rate = (h_colors - optimal_colors) / optimal_colors if optimal_colors > 0 else 0

            result.update({
                f'{method}_colors': h_colors,
                f'{method}_time': h_time,
                f'{method}_ratio': approx_ratio,
                f'{method}_error': error_rate,
                f'{method}_optimal': h_colors == optimal_colors
            })

        return result

    def comprehensive_quality_study(self, max_vertices: int = 9,
                                  samples_per_size: int = 20) -> pd.DataFrame:
        """
        Run comprehensive quality analysis across different graph types and sizes.

        Args:
            max_vertices: Maximum graph size (limited by brute force)
            samples_per_size: Number of random samples per size

        Returns:
            DataFrame with quality analysis results
        """
        print("Running comprehensive quality analysis...")
        print(f"Max vertices: {max_vertices}")
        print(f"Samples per size: {samples_per_size}")

        results = []

        for n_vertices in range(3, max_vertices + 1):
            print(f"Analyzing {n_vertices}-vertex graphs...", end=' ')

            # Test different graph types
            graph_configs = self._generate_graph_configurations(n_vertices, samples_per_size)

            for config in graph_configs:
                graph = config['graph']
                graph_info = {
                    'vertices': n_vertices,
                    'type': config['type'],
                    'density': config.get('density', 0)
                }

                result = self.analyze_single_graph(graph, graph_info)
                results.append(result)

            print(f"✓")

        self.results = pd.DataFrame(results)
        return self.results

    def _generate_graph_configurations(self, n_vertices: int, samples: int) -> List[Dict]:
        """Generate different graph configurations for testing."""
        configs = []

        # Complete graph
        if n_vertices <= 6:  # Complete graphs get expensive quickly
            complete_graph = generate_complete_graph(n_vertices)
            configs.append({
                'graph': complete_graph,
                'type': 'complete',
                'density': 1.0
            })

        # Cycle graph
        if n_vertices >= 3:
            cycle_graph = generate_cycle_graph(n_vertices)
            configs.append({
                'graph': cycle_graph,
                'type': 'cycle',
                'density': 2.0 / n_vertices
            })

        # Path graph
        if n_vertices >= 2:
            path_graph = Graph()
            for i in range(n_vertices):
                path_graph.add_vertex(i)
            for i in range(n_vertices - 1):
                path_graph.add_edge(i, i + 1)
            configs.append({
                'graph': path_graph,
                'type': 'path',
                'density': (n_vertices - 1) / (n_vertices * (n_vertices - 1) / 2)
            })

        # Random graphs with different densities
        max_edges = n_vertices * (n_vertices - 1) // 2
        densities = [0.2, 0.5, 0.8]  # Sparse, medium, dense

        for density in densities:
            n_edges = int(density * max_edges)
            for _ in range(samples // len(densities)):
                random_graph = generate_random_graph(n_vertices, n_edges)
                configs.append({
                    'graph': random_graph,
                    'type': f'random_{density:.1f}',
                    'density': density
                })

        return configs

    def analyze_by_graph_type(self) -> pd.DataFrame:
        """Analyze quality metrics grouped by graph type."""
        if self.results is None or len(self.results) == 0:
            print("No results available. Run quality analysis first.")
            return pd.DataFrame()

        methods = ['dsatur', 'ldf', 'hybrid']
        summary = []

        for graph_type in self.results['graph_type'].unique():
            type_data = self.results[self.results['graph_type'] == graph_type]

            for method in methods:
                ratio_col = f'{method}_ratio'
                error_col = f'{method}_error'
                optimal_col = f'{method}_optimal'

                if ratio_col in type_data.columns:
                    summary.append({
                        'graph_type': graph_type,
                        'method': method,
                        'mean_ratio': type_data[ratio_col].mean(),
                        'max_ratio': type_data[ratio_col].max(),
                        'mean_error': type_data[error_col].mean(),
                        'max_error': type_data[error_col].max(),
                        'optimal_solutions': type_data[optimal_col].sum(),
                        'total_tests': len(type_data),
                        'optimal_rate': type_data[optimal_col].mean()
                    })

        return pd.DataFrame(summary)

    def find_worst_cases(self, method: str = 'dsatur', metric: str = 'ratio',
                        top_n: int = 10) -> pd.DataFrame:
        """
        Find worst performing cases for a specific method.

        Args:
            method: Method to analyze
            metric: Metric to use ('ratio' or 'error')
            top_n: Number of worst cases to return

        Returns:
            DataFrame with worst cases
        """
        if self.results is None or len(self.results) == 0:
            print("No results available. Run quality analysis first.")
            return pd.DataFrame()

        metric_col = f'{method}_{metric}'
        if metric_col not in self.results.columns:
            print(f"Metric {metric_col} not found in results.")
            return pd.DataFrame()

        worst_cases = self.results.nlargest(top_n, metric_col)
        relevant_cols = ['vertices', 'edges', 'graph_type', 'optimal_colors',
                        f'{method}_colors', metric_col]

        return worst_cases[relevant_cols]

    def theoretical_bounds_analysis(self) -> Dict[str, Dict]:
        """
        Analyze how algorithms perform relative to known theoretical bounds.

        Returns:
            Dictionary with bounds analysis for each method
        """
        if self.results is None or len(self.results) == 0:
            return {}

        methods = ['dsatur', 'ldf', 'hybrid']
        bounds_analysis = {}

        for method in methods:
            ratio_col = f'{method}_ratio'
            ratios = self.results[ratio_col].dropna()

            # Calculate various metrics
            bounds_analysis[method] = {
                'mean_ratio': ratios.mean(),
                'median_ratio': ratios.median(),
                'max_ratio': ratios.max(),
                'std_ratio': ratios.std(),
                'worst_case_bound': ratios.max(),
                'percentile_95': ratios.quantile(0.95),
                'percentile_99': ratios.quantile(0.99),
                'optimal_solutions': (ratios == 1.0).sum(),
                'total_tests': len(ratios)
            }

            # Performance classification
            if bounds_analysis[method]['mean_ratio'] <= 1.1:
                bounds_analysis[method]['quality_class'] = 'Excellent (≤1.1 avg)'
            elif bounds_analysis[method]['mean_ratio'] <= 1.3:
                bounds_analysis[method]['quality_class'] = 'Good (≤1.3 avg)'
            elif bounds_analysis[method]['mean_ratio'] <= 1.5:
                bounds_analysis[method]['quality_class'] = 'Fair (≤1.5 avg)'
            else:
                bounds_analysis[method]['quality_class'] = 'Poor (>1.5 avg)'

        return bounds_analysis

    def visualize_quality_analysis(self, save_path: str = 'results/quality_analysis.png'):
        """
        Create comprehensive quality visualization.

        Args:
            save_path: Path to save the plot
        """
        if self.results is None or len(self.results) == 0:
            print("No results available for visualization.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Heuristic Quality Analysis', fontsize=16)

        methods = ['dsatur', 'ldf', 'hybrid']
        colors = {'dsatur': 'blue', 'ldf': 'green', 'hybrid': 'red'}

        # Plot 1: Approximation ratios by graph size
        ax1 = axes[0, 0]
        for method in methods:
            ratio_col = f'{method}_ratio'
            grouped = self.results.groupby('vertices')[ratio_col].mean()
            ax1.plot(grouped.index, grouped.values, 'o-',
                    label=method.upper(), color=colors[method])
        ax1.set_xlabel('Number of Vertices')
        ax1.set_ylabel('Mean Approximation Ratio')
        ax1.set_title('Approximation Ratio vs Graph Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Distribution of approximation ratios
        ax2 = axes[0, 1]
        for method in methods:
            ratio_col = f'{method}_ratio'
            ratios = self.results[ratio_col].dropna()
            ax2.hist(ratios, bins=20, alpha=0.7, label=method.upper(), color=colors[method])
        ax2.set_xlabel('Approximation Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Approximation Ratios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Quality by graph type
        ax3 = axes[1, 0]
        type_analysis = self.analyze_by_graph_type()
        if not type_analysis.empty:
            graph_types = type_analysis['graph_type'].unique()
            x_pos = np.arange(len(graph_types))
            width = 0.25

            for i, method in enumerate(methods):
                method_data = type_analysis[type_analysis['method'] == method]
                ratios = [method_data[method_data['graph_type'] == gt]['mean_ratio'].iloc[0]
                         if len(method_data[method_data['graph_type'] == gt]) > 0 else 0
                         for gt in graph_types]
                ax3.bar(x_pos + i*width, ratios, width, label=method.upper(), color=colors[method])

            ax3.set_xlabel('Graph Type')
            ax3.set_ylabel('Mean Approximation Ratio')
            ax3.set_title('Quality by Graph Type')
            ax3.set_xticks(x_pos + width)
            ax3.set_xticklabels(graph_types, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Optimal solution rates
        ax4 = axes[1, 1]
        optimal_rates = []
        method_names = []
        for method in methods:
            optimal_col = f'{method}_optimal'
            rate = self.results[optimal_col].mean() * 100
            optimal_rates.append(rate)
            method_names.append(method.upper())

        bars = ax4.bar(method_names, optimal_rates, color=[colors[m.lower()] for m in method_names])
        ax4.set_ylabel('Optimal Solutions (%)')
        ax4.set_title('Rate of Optimal Solutions')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Quality analysis visualization saved to {save_path}")

    def generate_quality_report(self) -> str:
        """Generate comprehensive quality analysis report."""
        if self.results is None or len(self.results) == 0:
            return "No results available for quality report."

        report = []
        report.append("="*80)
        report.append("HEURISTIC QUALITY ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Total tests performed: {len(self.results)}")
        report.append(f"Graph size range: {self.results['vertices'].min()} - {self.results['vertices'].max()} vertices")
        report.append(f"Graph types: {', '.join(self.results['graph_type'].unique())}")
        report.append("")

        # Theoretical bounds analysis
        bounds_analysis = self.theoretical_bounds_analysis()

        for method, metrics in bounds_analysis.items():
            report.append(f"\n{method.upper()} ALGORITHM QUALITY:")
            report.append("-" * 40)
            report.append(f"Mean approximation ratio: {metrics['mean_ratio']:.3f}")
            report.append(f"Worst-case ratio: {metrics['max_ratio']:.3f}")
            report.append(f"Standard deviation: {metrics['std_ratio']:.3f}")
            report.append(f"95th percentile: {metrics['percentile_95']:.3f}")
            report.append(f"Optimal solutions: {metrics['optimal_solutions']}/{metrics['total_tests']} ({metrics['optimal_solutions']/metrics['total_tests']*100:.1f}%)")
            report.append(f"Quality classification: {metrics['quality_class']}")

        # Graph type analysis
        report.append(f"\nQUALITY BY GRAPH TYPE:")
        report.append("-" * 30)
        type_analysis = self.analyze_by_graph_type()

        for graph_type in type_analysis['graph_type'].unique():
            report.append(f"\n{graph_type.upper()}:")
            type_data = type_analysis[type_analysis['graph_type'] == graph_type]
            for _, row in type_data.iterrows():
                report.append(f"  {row['method'].upper()}: {row['mean_ratio']:.3f} ratio, {row['optimal_rate']*100:.1f}% optimal")

        return '\n'.join(report)

    def save_results(self, filename: str = 'results/quality_analysis.csv'):
        """Save quality analysis results to CSV."""
        if self.results is not None and len(self.results) > 0:
            self.results.to_csv(filename, index=False)
            print(f"Quality analysis results saved to {filename}")
        else:
            print("No results to save.")