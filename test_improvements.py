"""
Test script to evaluate improved heuristic algorithms.
"""

import sys
from pathlib import Path
sys.path.append("src")

from benchmark import BenchmarkSuite
import pandas as pd

def main():
    print("="*60)
    print("IMPROVED HEURISTIC ALGORITHMS TEST")
    print("="*60)

    # Create benchmark suite
    benchmark = BenchmarkSuite()

    # Run comprehensive accuracy analysis with new methods
    print("\n1. Running accuracy analysis with improved algorithms...")
    results_df = benchmark.run_accuracy_analysis(max_vertices=8, samples_per_size=5)

    print(f"\nTotal tests run: {len(results_df)}")

    # Overall method comparison
    print("\n2. COMPARISON: Original vs Improved Methods:")
    comparison = benchmark.compare_methods_overall()

    # Sort methods by performance
    sorted_methods = sorted(comparison.items(),
                           key=lambda x: x[1]['mean_error_rate'])

    print(f"\n{'Method':<15} {'Error Rate':<12} {'Failure Rate':<12} {'Perfect Solutions':<15}")
    print("-" * 60)
    for method, stats in sorted_methods:
        print(f"{method.upper():<15} "
              f"{stats['mean_error_rate']*100:.1f}%{'':<8} "
              f"{stats['failure_rate']*100:.1f}%{'':<8} "
              f"{stats['perfect_solutions']}/{stats['total_tests']}")

    # Focus on problem areas (sparse random graphs)
    print("\n3. PERFORMANCE ON SPARSE RANDOM GRAPHS (Problem Area):")
    sparse_data = results_df[results_df['graph_type'] == 'sparse_random']

    if len(sparse_data) > 0:
        print(f"\nSparse random graphs tested: {len(sparse_data)}")

        methods_to_check = ['dsatur', 'ldf', 'hybrid']
        for method in methods_to_check:
            error_col = f'{method}_error_rate'
            if error_col in sparse_data.columns:
                mean_error = sparse_data[error_col].mean()
                max_error = sparse_data[error_col].max()
                failures = (sparse_data[error_col] > 0).sum()
                print(f"{method.upper():<15}: {mean_error*100:.1f}% avg, {max_error*100:.1f}% max, {failures} failures")

    # Detailed worst cases
    print("\n4. WORST CASES ANALYSIS:")

    # Check worst cases for main algorithms
    worst_cases_dsatur = benchmark.find_worst_cases('dsatur', top_n=3)
    if not worst_cases_dsatur.empty:
        print("\nWorst cases for DSATUR:")
        print(worst_cases_dsatur[['vertices', 'edges', 'graph_type', 'optimal_colors', 'dsatur_colors', 'dsatur_error_rate']].to_string(index=False))

    if 'hybrid_error_rate' in results_df.columns:
        worst_cases_hybrid = benchmark.find_worst_cases('hybrid', top_n=3)
        if not worst_cases_hybrid.empty:
            print("\nWorst cases for HYBRID:")
            print(worst_cases_hybrid[['vertices', 'edges', 'graph_type', 'optimal_colors', 'hybrid_colors', 'hybrid_error_rate']].to_string(index=False))

    # Speed vs Quality analysis
    print("\n5. SPEED vs QUALITY ANALYSIS:")
    if len(comparison) > 0:
        print(f"\n{'Method':<15} {'Avg Time (ms)':<15} {'Error Rate':<12} {'Quality/Speed'}")
        print("-" * 60)
        for method, stats in sorted_methods:
            time_ms = stats['mean_time'] * 1000
            quality_speed = stats['mean_error_rate'] / stats['mean_time'] if stats['mean_time'] > 0 else 0
            print(f"{method.upper():<15} {time_ms:.3f}{'':<10} {stats['mean_error_rate']*100:.1f}%{'':<8} {quality_speed:.3f}")

    # Final recommendations
    print("\n6. IMPROVEMENT RESULTS:")
    best_method = min(comparison.keys(), key=lambda m: comparison[m]['mean_error_rate'])
    print(f"✅ Best performing method: {best_method.upper()}")

    if comparison[best_method]['mean_error_rate'] < 0.05:
        print("✅ SUCCESS: Error rate < 5% achieved!")
    else:
        print(f"⚠️  Current error rate: {comparison[best_method]['mean_error_rate']*100:.1f}%")
        print("   Consider implementing meta-heuristic algorithms for further improvement.")

    # Save results
    benchmark.save_results("results/improved_benchmark_results.csv")
    print(f"\n📊 Detailed results saved to results/improved_benchmark_results.csv")

if __name__ == "__main__":
    main()