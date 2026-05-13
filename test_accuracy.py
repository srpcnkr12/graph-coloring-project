"""
Test script to analyze current heuristic accuracy.
"""

import sys
from pathlib import Path
sys.path.append("src")

from benchmark import BenchmarkSuite
import pandas as pd

def main():
    print("="*60)
    print("HEURISTIC ACCURACY ANALYSIS")
    print("="*60)

    # Create benchmark suite
    benchmark = BenchmarkSuite()

    # Run comprehensive accuracy analysis
    print("\n1. Running accuracy analysis...")
    results_df = benchmark.run_accuracy_analysis(max_vertices=8, samples_per_size=5)

    print(f"\nTotal tests run: {len(results_df)}")
    print(f"Graph types tested: {results_df['graph_type'].unique()}")

    # Overall method comparison
    print("\n2. Overall Method Comparison:")
    comparison = benchmark.compare_methods_overall()

    for method, stats in comparison.items():
        print(f"\n{method.upper()}:")
        print(f"  Mean error rate: {stats['mean_error_rate']:.3f} ({stats['mean_error_rate']*100:.1f}%)")
        print(f"  Failure rate: {stats['failure_rate']:.3f} ({stats['failure_rate']*100:.1f}%)")
        print(f"  Perfect solutions: {stats['perfect_solutions']}/{stats['total_tests']}")
        print(f"  Max error: {stats['max_error_rate']:.3f}")

    # Analysis by graph type
    print("\n3. Performance by Graph Type:")
    type_analysis = benchmark.analyze_accuracy_by_graph_type()

    for graph_type in type_analysis['graph_type'].unique():
        print(f"\n{graph_type.upper()}:")
        type_data = type_analysis[type_analysis['graph_type'] == graph_type]
        for _, row in type_data.iterrows():
            print(f"  {row['method']}: {row['mean_error_rate']:.3f} error rate")

    # Find worst cases
    print("\n4. Worst Cases (DSATUR Algorithm):")
    worst_cases = benchmark.find_worst_cases('dsatur', top_n=5)
    print(worst_cases.to_string(index=False))

    # Recommendations
    print("\n5. Improvement Recommendations:")
    recommendations = benchmark.generate_improvement_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Save results
    benchmark.save_results()
    print(f"\nDetailed results saved to results/benchmark_results.csv")

if __name__ == "__main__":
    main()