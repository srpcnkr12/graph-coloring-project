"""
Comprehensive quality analysis testing script.
Evaluates solution quality of heuristic algorithms compared to optimal solutions.
"""

import sys
from pathlib import Path
sys.path.append("src")

from quality_analysis import QualityAnalyzer
import pandas as pd

def main():
    print("="*80)
    print("HEURISTIC QUALITY ANALYSIS")
    print("="*80)
    print("Evaluating solution quality compared to optimal solutions")
    print("Analyzing approximation ratios and quality metrics")
    print("")

    # Initialize quality analyzer
    analyzer = QualityAnalyzer()

    # Run comprehensive quality study
    print("1. COMPREHENSIVE QUALITY STUDY")
    print("-" * 40)
    print("Testing graphs up to 9 vertices (brute force limit)")
    print("Analyzing different graph types and densities")

    results_df = analyzer.comprehensive_quality_study(
        max_vertices=9,
        samples_per_size=15
    )

    print(f"\nTotal quality tests completed: {len(results_df)}")

    # Overall quality summary
    print("\n2. OVERALL QUALITY SUMMARY")
    print("-" * 40)

    methods = ['dsatur', 'ldf', 'hybrid']
    print(f"{'Method':<10} {'Mean Ratio':<12} {'Max Ratio':<12} {'Optimal %':<12} {'Quality'}")
    print("-" * 60)

    for method in methods:
        ratio_col = f'{method}_ratio'
        optimal_col = f'{method}_optimal'

        mean_ratio = results_df[ratio_col].mean()
        max_ratio = results_df[ratio_col].max()
        optimal_rate = results_df[optimal_col].mean() * 100

        if mean_ratio <= 1.1:
            quality = "Excellent"
        elif mean_ratio <= 1.3:
            quality = "Good"
        elif mean_ratio <= 1.5:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"{method.upper():<10} {mean_ratio:<12.3f} {max_ratio:<12.3f} {optimal_rate:<12.1f} {quality}")

    # Quality by graph type
    print("\n3. QUALITY BY GRAPH TYPE")
    print("-" * 40)

    type_analysis = analyzer.analyze_by_graph_type()
    if not type_analysis.empty:
        print(f"{'Graph Type':<15} {'Method':<10} {'Mean Ratio':<12} {'Optimal %':<12}")
        print("-" * 55)

        for graph_type in sorted(type_analysis['graph_type'].unique()):
            type_data = type_analysis[type_analysis['graph_type'] == graph_type]
            for _, row in type_data.iterrows():
                print(f"{graph_type:<15} {row['method'].upper():<10} "
                      f"{row['mean_ratio']:<12.3f} {row['optimal_rate']*100:<12.1f}")

    # Theoretical bounds analysis
    print("\n4. THEORETICAL BOUNDS ANALYSIS")
    print("-" * 40)

    bounds = analyzer.theoretical_bounds_analysis()
    for method, metrics in bounds.items():
        print(f"\n{method.upper()} Algorithm:")
        print(f"  Average approximation ratio: {metrics['mean_ratio']:.3f}")
        print(f"  Worst-case bound: {metrics['worst_case_bound']:.3f}")
        print(f"  95th percentile: {metrics['percentile_95']:.3f}")
        print(f"  Optimal solutions: {metrics['optimal_solutions']}/{metrics['total_tests']} "
              f"({metrics['optimal_solutions']/metrics['total_tests']*100:.1f}%)")
        print(f"  Quality class: {metrics['quality_class']}")

    # Worst cases analysis
    print("\n5. WORST CASES ANALYSIS")
    print("-" * 40)

    for method in methods:
        print(f"\nWorst cases for {method.upper()} (by approximation ratio):")
        worst_cases = analyzer.find_worst_cases(method, metric='ratio', top_n=5)
        if not worst_cases.empty:
            print(worst_cases.to_string(index=False, float_format='%.3f'))
        else:
            print("No worst cases found.")

    # Performance vs Quality trade-off
    print("\n6. PERFORMANCE vs QUALITY TRADE-OFF")
    print("-" * 40)

    print(f"{'Method':<10} {'Quality (Ratio)':<15} {'Speed Ranking':<15} {'Overall'}")
    print("-" * 50)

    # Simple ranking based on mean ratio (lower is better)
    quality_ranking = {}
    for method in methods:
        ratio_col = f'{method}_ratio'
        quality_ranking[method] = results_df[ratio_col].mean()

    sorted_by_quality = sorted(quality_ranking.items(), key=lambda x: x[1])

    # Speed ranking (based on typical complexity knowledge)
    speed_ranking = {
        'ldf': 1,      # O(n²) but simple
        'dsatur': 2,   # O(n³) more complex
        'hybrid': 3    # Runs multiple algorithms
    }

    for i, (method, ratio) in enumerate(sorted_by_quality, 1):
        speed_rank = speed_ranking[method]
        overall_score = i + speed_rank  # Lower is better
        print(f"{method.upper():<10} #{i} ({ratio:.3f})<6} #{speed_rank} (complexity)<9} #{overall_score}")

    # Statistical significance
    print("\n7. STATISTICAL SIGNIFICANCE")
    print("-" * 40)

    total_tests = len(results_df)
    print(f"Total tests performed: {total_tests}")
    print(f"Graph sizes tested: {results_df['vertices'].min()}-{results_df['vertices'].max()} vertices")
    print(f"Graph types: {len(results_df['graph_type'].unique())} different types")

    # Check for statistical significance
    for method in methods:
        ratio_col = f'{method}_ratio'
        ratios = results_df[ratio_col]
        mean_ratio = ratios.mean()
        std_ratio = ratios.std()
        n = len(ratios)

        # 95% confidence interval for mean
        margin_error = 1.96 * std_ratio / (n ** 0.5)
        ci_lower = mean_ratio - margin_error
        ci_upper = mean_ratio + margin_error

        print(f"\n{method.upper()} 95% confidence interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        if ci_upper < 1.2:
            print(f"  ✅ Statistically excellent (CI upper bound < 1.2)")
        elif ci_upper < 1.5:
            print(f"  ✅ Statistically good (CI upper bound < 1.5)")
        else:
            print(f"  ⚠️  Statistical quality needs improvement")

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Save results
    print("\n8. SAVING RESULTS")
    print("-" * 40)

    analyzer.save_results("results/quality_analysis.csv")

    # Generate and save report
    report = analyzer.generate_quality_report()
    with open("results/quality_report.txt", "w") as f:
        f.write(report)
    print("Quality report saved to results/quality_report.txt")

    # Create visualization
    print("Generating quality visualization...")
    analyzer.visualize_quality_analysis("results/quality_plots.png")

    print("\n" + "="*80)
    print("QUALITY ANALYSIS COMPLETE")
    print("="*80)
    print("Files generated:")
    print("- results/quality_analysis.csv")
    print("- results/quality_report.txt")
    print("- results/quality_plots.png")

    # Final recommendation
    print("\nRECOMMENDATION:")
    print("-" * 15)

    best_overall = sorted_by_quality[0][0]  # Best quality method
    print(f"Best overall algorithm: {best_overall.upper()}")

    best_ratio = quality_ranking[best_overall]
    if best_ratio <= 1.1:
        print(f"✅ EXCELLENT quality achieved (avg ratio: {best_ratio:.3f})")
        print("   Ready for academic submission.")
    elif best_ratio <= 1.3:
        print(f"✅ GOOD quality achieved (avg ratio: {best_ratio:.3f})")
        print("   Meets academic standards.")
    else:
        print(f"⚠️  Quality could be improved (avg ratio: {best_ratio:.3f})")
        print("   Consider algorithm refinements.")

if __name__ == "__main__":
    main()