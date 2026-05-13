"""
Statistical performance testing module with confidence intervals.
Implements rigorous performance analysis required for academic projects.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    from .graph import Graph, generate_random_graph
    from .heuristic import HeuristicColoring
except ImportError:
    from graph import Graph, generate_random_graph
    from heuristic import HeuristicColoring


class PerformanceTester:
    """
    Statistical performance testing with confidence intervals.
    Implements requirements for academic performance analysis.
    """

    def __init__(self, confidence_level: float = 0.90):
        """
        Initialize performance tester.

        Args:
            confidence_level: Statistical confidence level (default 0.90 for 90%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results = []

    def measure_execution_time(self, func: Callable, *args, **kwargs) -> float:
        """
        Precisely measure execution time of a function.

        Args:
            func: Function to measure
            *args, **kwargs: Arguments for the function

        Returns:
            Execution time in seconds
        """
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time

    def run_multiple_measurements(self, func: Callable, n_runs: int,
                                 *args, **kwargs) -> List[float]:
        """
        Run multiple measurements for statistical analysis.

        Args:
            func: Function to measure
            n_runs: Number of measurements
            *args, **kwargs: Arguments for the function

        Returns:
            List of execution times
        """
        times = []
        for _ in range(n_runs):
            exec_time = self.measure_execution_time(func, *args, **kwargs)
            times.append(exec_time)
        return times

    def calculate_confidence_interval(self, measurements: List[float]) -> Tuple[float, float, float, float]:
        """
        Calculate confidence interval for measurements.

        Args:
            measurements: List of execution time measurements

        Returns:
            Tuple of (mean, std_error, lower_bound, upper_bound)
        """
        n = len(measurements)
        mean = np.mean(measurements)
        std_dev = np.std(measurements, ddof=1)  # Sample standard deviation
        std_error = std_dev / np.sqrt(n)

        # Calculate t-critical value for confidence interval
        df = n - 1
        t_critical = stats.t.ppf(1 - self.alpha/2, df)

        margin_error = t_critical * std_error
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error

        return mean, std_error, lower_bound, upper_bound

    def is_interval_narrow_enough(self, mean: float, margin_error: float,
                                 threshold: float = 0.1) -> bool:
        """
        Check if confidence interval is narrow enough (b/a < threshold).

        Args:
            mean: Mean of measurements
            margin_error: Half-width of confidence interval
            threshold: Maximum acceptable ratio (default 0.1)

        Returns:
            True if interval is narrow enough
        """
        if mean <= 0:
            return False
        ratio = margin_error / mean
        return ratio < threshold

    def performance_analysis(self, method: str, vertex_sizes: List[int],
                           n_runs: int = 30, edge_density: float = 0.3) -> pd.DataFrame:
        """
        Perform statistical performance analysis for a heuristic method.

        Args:
            method: Heuristic method name ('dsatur', 'ldf', 'hybrid')
            vertex_sizes: List of graph sizes to test
            n_runs: Number of runs per size for statistical significance
            edge_density: Density of test graphs

        Returns:
            DataFrame with performance analysis results
        """
        print(f"Running performance analysis for {method.upper()}...")
        print(f"Confidence level: {self.confidence_level*100:.0f}%")
        print(f"Runs per size: {n_runs}")
        print(f"Edge density: {edge_density:.1f}")

        results = []

        for n_vertices in vertex_sizes:
            print(f"Testing size {n_vertices}...", end=' ')

            # Generate test graph
            n_edges = int(edge_density * n_vertices * (n_vertices - 1) / 2)
            graph = generate_random_graph(n_vertices, n_edges)
            heuristic = HeuristicColoring(graph)

            # Run multiple measurements
            measurements = self.run_multiple_measurements(
                heuristic.solve, n_runs, method
            )
            # Calculate statistics
            mean, std_error, lower_bound, upper_bound = self.calculate_confidence_interval(measurements)
            margin_error = upper_bound - mean
            is_narrow = self.is_interval_narrow_enough(mean, margin_error)

            result = {
                'vertices': n_vertices,
                'edges': graph.num_edges,
                'method': method,
                'n_runs': n_runs,
                'mean_time': mean,
                'std_error': std_error,
                'confidence_lower': lower_bound,
                'confidence_upper': upper_bound,
                'margin_error': margin_error,
                'relative_error': margin_error / mean if mean > 0 else float('inf'),
                'is_narrow_enough': is_narrow,
                'confidence_level': self.confidence_level
            }

            results.append(result)
            print(f"Mean: {mean:.6f}s, Relative error: {result['relative_error']:.3f}")

        return pd.DataFrame(results)

    def comprehensive_performance_study(self, max_vertices: int = 1000,
                                      step_size: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive performance study for all heuristic methods.

        Args:
            max_vertices: Maximum graph size to test
            step_size: Step size between graph sizes

        Returns:
            Dictionary mapping method names to their performance DataFrames
        """
        methods = ['dsatur', 'ldf', 'hybrid']
        vertex_sizes = list(range(50, max_vertices + 1, step_size))

        results = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"PERFORMANCE ANALYSIS: {method.upper()}")
            print(f"{'='*60}")

            df = self.performance_analysis(method, vertex_sizes)
            results[method] = df

            # Check if all intervals are narrow enough
            narrow_count = df['is_narrow_enough'].sum()
            total_count = len(df)
            print(f"Narrow intervals: {narrow_count}/{total_count} ({narrow_count/total_count*100:.1f}%)")

        return results

    def fit_complexity_curve(self, df: pd.DataFrame, method: str) -> Dict[str, float]:
        """
        Fit complexity curve to performance measurements.

        Args:
            df: Performance measurements DataFrame
            method: Method name

        Returns:
            Dictionary with curve fitting results
        """
        x = df['vertices'].values
        y = df['mean_time'].values

        # Try different curve fits
        fits = {}

        # Linear: T(n) = an + b
        try:
            coeffs_linear = np.polyfit(x, y, 1)
            y_pred_linear = np.polyval(coeffs_linear, x)
            r2_linear = 1 - (np.sum((y - y_pred_linear) ** 2) / np.sum((y - np.mean(y)) ** 2))
            fits['linear'] = {'coefficients': coeffs_linear, 'r2': r2_linear, 'equation': f'{coeffs_linear[0]:.2e}*n + {coeffs_linear[1]:.2e}'}
        except:
            fits['linear'] = {'r2': 0, 'equation': 'Failed to fit'}

        # Quadratic: T(n) = an² + bn + c
        try:
            coeffs_quad = np.polyfit(x, y, 2)
            y_pred_quad = np.polyval(coeffs_quad, x)
            r2_quad = 1 - (np.sum((y - y_pred_quad) ** 2) / np.sum((y - np.mean(y)) ** 2))
            fits['quadratic'] = {'coefficients': coeffs_quad, 'r2': r2_quad, 'equation': f'{coeffs_quad[0]:.2e}*n² + {coeffs_quad[1]:.2e}*n + {coeffs_quad[2]:.2e}'}
        except:
            fits['quadratic'] = {'r2': 0, 'equation': 'Failed to fit'}

        # Log-linear: T(n) = a*n*log(n) + b
        try:
            x_nlogn = x * np.log2(x)
            X_matrix = np.column_stack([x_nlogn, np.ones(len(x))])
            coeffs_nlogn = np.linalg.lstsq(X_matrix, y, rcond=None)[0]
            y_pred_nlogn = X_matrix @ coeffs_nlogn
            r2_nlogn = 1 - (np.sum((y - y_pred_nlogn) ** 2) / np.sum((y - np.mean(y)) ** 2))
            fits['nlogn'] = {'coefficients': coeffs_nlogn, 'r2': r2_nlogn, 'equation': f'{coeffs_nlogn[0]:.2e}*n*log₂(n) + {coeffs_nlogn[1]:.2e}'}
        except:
            fits['nlogn'] = {'r2': 0, 'equation': 'Failed to fit'}

        return fits

    def visualize_performance(self, results: Dict[str, pd.DataFrame],
                            save_path: str = 'results/performance_analysis.png'):
        """
        Create comprehensive performance visualization.

        Args:
            results: Dictionary mapping method names to performance DataFrames
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Performance Analysis with Confidence Intervals', fontsize=16)

        colors = {'dsatur': 'blue', 'ldf': 'green', 'hybrid': 'red'}

        # Plot 1: Execution time with confidence intervals
        ax1 = axes[0, 0]
        for method, df in results.items():
            ax1.errorbar(df['vertices'], df['mean_time'],
                        yerr=df['margin_error'],
                        label=method.upper(), color=colors[method],
                        capsize=3, alpha=0.8)
        ax1.set_xlabel('Number of Vertices')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Graph Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Log-log scale
        ax2 = axes[0, 1]
        for method, df in results.items():
            ax2.loglog(df['vertices'], df['mean_time'],
                      'o-', label=method.upper(), color=colors[method])
        ax2.set_xlabel('Number of Vertices (log scale)')
        ax2.set_ylabel('Execution Time (log scale)')
        ax2.set_title('Log-Log Performance Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Relative error (b/a ratios)
        ax3 = axes[1, 0]
        for method, df in results.items():
            ax3.plot(df['vertices'], df['relative_error'],
                    'o-', label=method.upper(), color=colors[method])
        ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
        ax3.set_xlabel('Number of Vertices')
        ax3.set_ylabel('Relative Error (b/a)')
        ax3.set_title('Confidence Interval Quality')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Curve fitting comparison
        ax4 = axes[1, 1]
        # Show best fitting curves
        for method, df in results.items():
            fits = self.fit_complexity_curve(df, method)
            best_fit = max(fits.items(), key=lambda x: x[1]['r2'] if isinstance(x[1]['r2'], (int, float)) else 0)

            x = df['vertices'].values
            if best_fit[0] == 'linear':
                y_pred = np.polyval(fits['linear']['coefficients'], x)
            elif best_fit[0] == 'quadratic':
                y_pred = np.polyval(fits['quadratic']['coefficients'], x)
            elif best_fit[0] == 'nlogn':
                X_matrix = np.column_stack([x * np.log2(x), np.ones(len(x))])
                y_pred = X_matrix @ fits['nlogn']['coefficients']
            else:
                continue

            ax4.plot(x, y_pred, '--', color=colors[method], alpha=0.8,
                    label=f'{method.upper()} ({best_fit[0]}, R²={best_fit[1]["r2"]:.3f})')
            ax4.scatter(df['vertices'], df['mean_time'], color=colors[method], alpha=0.6, s=20)

        ax4.set_xlabel('Number of Vertices')
        ax4.set_ylabel('Execution Time (seconds)')
        ax4.set_title('Best Fitting Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance visualization saved to {save_path}")

    def generate_performance_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate detailed performance analysis report.

        Args:
            results: Dictionary mapping method names to performance DataFrames

        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("STATISTICAL PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Confidence Level: {self.confidence_level*100:.0f}%")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for method, df in results.items():
            report.append(f"\n{method.upper()} ALGORITHM ANALYSIS:")
            report.append("-" * 40)

            # Statistical summary
            mean_times = df['mean_time']
            report.append(f"Graph size range: {df['vertices'].min()} - {df['vertices'].max()} vertices")
            report.append(f"Total measurements: {df['n_runs'].iloc[0] * len(df)}")
            report.append(f"Mean execution time range: {mean_times.min():.6f} - {mean_times.max():.6f} seconds")

            # Confidence interval quality
            narrow_count = df['is_narrow_enough'].sum()
            total_count = len(df)
            report.append(f"Narrow confidence intervals: {narrow_count}/{total_count} ({narrow_count/total_count*100:.1f}%)")

            # Complexity analysis
            fits = self.fit_complexity_curve(df, method)
            best_fit = max(fits.items(), key=lambda x: x[1]['r2'] if isinstance(x[1]['r2'], (int, float)) else 0)
            report.append(f"Best fitting curve: {best_fit[0]} (R² = {best_fit[1]['r2']:.4f})")
            report.append(f"Equation: T(n) = {best_fit[1]['equation']}")

            # Performance summary
            if best_fit[0] == 'linear':
                report.append("Complexity class: O(n) - Linear")
            elif best_fit[0] == 'quadratic':
                report.append("Complexity class: O(n²) - Quadratic")
            elif best_fit[0] == 'nlogn':
                report.append("Complexity class: O(n log n) - Linearithmic")

        return '\n'.join(report)

    def save_results(self, results: Dict[str, pd.DataFrame],
                    filename: str = 'results/performance_results.csv'):
        """
        Save performance results to CSV.

        Args:
            results: Dictionary mapping method names to performance DataFrames
            filename: Output filename
        """
        all_results = []
        for method, df in results.items():
            all_results.append(df)

        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(filename, index=False)
        print(f"Performance results saved to {filename}")