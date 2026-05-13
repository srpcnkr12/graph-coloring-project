"""
Comprehensive performance testing with statistical analysis.
Implements academic-grade performance evaluation with confidence intervals.
"""

import sys
from pathlib import Path
sys.path.append("src")

from performance_testing import PerformanceTester
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("STATISTICAL PERFORMANCE ANALYSIS")
    print("="*80)
    print("Academic-grade performance evaluation with confidence intervals")
    print("Confidence level: 90%")
    print("Requirements: b/a < 0.1 for narrow intervals")
    print("")

    # Initialize performance tester with 90% confidence
    tester = PerformanceTester(confidence_level=0.90)

    # Test sizes for comprehensive analysis
    small_sizes = list(range(20, 200, 20))   # Small graphs for detailed analysis
    medium_sizes = list(range(200, 600, 50)) # Medium graphs for scaling
    large_sizes = list(range(600, 1200, 100)) # Large graphs for performance limits

    print("1. SMALL GRAPH PERFORMANCE (20-180 vertices)")
    print("-" * 50)

    small_results = {}
    for method in ['dsatur', 'ldf', 'hybrid']:
        print(f"\nTesting {method.upper()}...")
        df = tester.performance_analysis(method, small_sizes, n_runs=20)
        small_results[method] = df

        # Quality check for confidence intervals
        narrow_count = df['is_narrow_enough'].sum()
        total_count = len(df)
        print(f"Narrow intervals: {narrow_count}/{total_count} ({narrow_count/total_count*100:.1f}%)")

    print("\n" + "="*60)
    print("2. MEDIUM GRAPH PERFORMANCE (200-550 vertices)")
    print("-" * 50)

    medium_results = {}
    for method in ['dsatur', 'ldf', 'hybrid']:
        print(f"\nTesting {method.upper()}...")
        df = tester.performance_analysis(method, medium_sizes, n_runs=15)
        medium_results[method] = df

    print("\n" + "="*60)
    print("3. LARGE GRAPH PERFORMANCE (600-1100 vertices)")
    print("-" * 50)

    large_results = {}
    for method in ['dsatur', 'ldf', 'hybrid']:
        print(f"\nTesting {method.upper()}...")
        df = tester.performance_analysis(method, large_sizes, n_runs=10)
        large_results[method] = df

    # Combine all results
    print("\n" + "="*60)
    print("4. COMPREHENSIVE ANALYSIS")
    print("-" * 50)

    all_results = {}
    for method in ['dsatur', 'ldf', 'hybrid']:
        combined_df = pd.concat([
            small_results[method],
            medium_results[method],
            large_results[method]
        ], ignore_index=True)
        all_results[method] = combined_df

    # Generate complexity analysis
    print("\n5. COMPLEXITY ANALYSIS:")
    print("-" * 30)

    for method, df in all_results.items():
        print(f"\n{method.upper()} Algorithm:")

        # Fit complexity curves
        fits = tester.fit_complexity_curve(df, method)
        best_fit = max(fits.items(), key=lambda x: x[1]['r2'] if isinstance(x[1]['r2'], (int, float)) else 0)

        print(f"  Best fitting curve: {best_fit[0]}")
        print(f"  R² = {best_fit[1]['r2']:.4f}")
        print(f"  Equation: T(n) = {best_fit[1]['equation']}")

        # Performance summary
        min_time = df['mean_time'].min()
        max_time = df['mean_time'].max()
        print(f"  Time range: {min_time:.6f} - {max_time:.6f} seconds")

        # Scalability assessment
        if best_fit[1]['r2'] > 0.9:
            if best_fit[0] == 'linear':
                print(f"  Complexity: O(n) - Excellent scalability")
            elif best_fit[0] == 'nlogn':
                print(f"  Complexity: O(n log n) - Good scalability")
            elif best_fit[0] == 'quadratic':
                print(f"  Complexity: O(n²) - Limited scalability")

    # Statistical summary
    print("\n6. STATISTICAL SUMMARY:")
    print("-" * 30)

    for method, df in all_results.items():
        total_measurements = df['n_runs'].sum()
        narrow_intervals = df['is_narrow_enough'].sum()
        total_intervals = len(df)
        mean_relative_error = df['relative_error'].mean()

        print(f"\n{method.upper()}:")
        print(f"  Total measurements: {total_measurements}")
        print(f"  Narrow intervals: {narrow_intervals}/{total_intervals} ({narrow_intervals/total_intervals*100:.1f}%)")
        print(f"  Mean relative error: {mean_relative_error:.3f}")

        if narrow_intervals/total_intervals >= 0.8:
            print(f"  ✅ Statistical quality: EXCELLENT")
        elif narrow_intervals/total_intervals >= 0.6:
            print(f"  ⚠️  Statistical quality: GOOD")
        else:
            print(f"  ❌ Statistical quality: POOR")

    # Performance ranking
    print("\n7. PERFORMANCE RANKING:")
    print("-" * 30)

    # Calculate average performance for largest graphs
    large_graph_performance = []
    largest_size = max([df['vertices'].max() for df in all_results.values()])

    for method, df in all_results.items():
        largest_df = df[df['vertices'] == largest_size]
        if len(largest_df) > 0:
            avg_time = largest_df['mean_time'].iloc[0]
            large_graph_performance.append((method, avg_time))

    large_graph_performance.sort(key=lambda x: x[1])

    print(f"Performance on {largest_size}-vertex graphs:")
    for i, (method, time) in enumerate(large_graph_performance, 1):
        print(f"{i}. {method.upper()}: {time:.6f} seconds")

    # Save results
    print("\n8. SAVING RESULTS:")
    print("-" * 30)

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Save performance data
    tester.save_results(all_results, "results/performance_analysis.csv")

    # Generate and save report
    report = tester.generate_performance_report(all_results)
    with open("results/performance_report.txt", "w") as f:
        f.write(report)
    print("Performance report saved to results/performance_report.txt")

    # Create visualization
    print("Generating performance visualization...")
    tester.visualize_performance(all_results, "results/performance_plots.png")

    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("="*80)
    print("Files generated:")
    print("- results/performance_analysis.csv")
    print("- results/performance_report.txt")
    print("- results/performance_plots.png")

if __name__ == "__main__":
    main()