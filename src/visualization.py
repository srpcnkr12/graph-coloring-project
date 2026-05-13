"""
Visualization module for graph coloring results.
Provides functions to plot graphs with their colorings and performance comparisons.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import random
from typing import Dict, List, Optional
try:
    from .graph import Graph
except ImportError:
    from graph import Graph


class GraphVisualizer:
    """
    Class for visualizing graph coloring results.
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
        random.shuffle(self.colors)

    def plot_graph_coloring(self, graph: Graph, coloring: Dict[int, int],
                          title: str = "Graph Coloring", save_path: Optional[str] = None,
                          figsize: tuple = (10, 8)) -> None:
        """
        Plot a graph with its coloring.

        Args:
            graph: The graph to visualize
            coloring: Dictionary mapping vertices to colors
            title: Title for the plot
            save_path: Path to save the figure (optional)
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)

        # Get the NetworkX graph
        nx_graph = graph.to_networkx()

        # Create color map for vertices
        unique_colors = list(set(coloring.values()))
        color_map = {color: self.colors[i % len(self.colors)] for i, color in enumerate(unique_colors)}
        node_colors = [color_map[coloring[node]] for node in nx_graph.nodes()]

        # Choose layout based on graph size
        if len(nx_graph.nodes()) <= 20:
            pos = nx.spring_layout(nx_graph, k=3, iterations=100)
        else:
            pos = nx.spring_layout(nx_graph, k=1, iterations=50)

        # Draw the graph
        nx.draw(nx_graph, pos,
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_size=10,
                font_weight='bold',
                edge_color='gray',
                width=2)

        # Add title and color information
        num_colors = len(unique_colors)
        plt.title(f"{title}\nVertices: {graph.num_vertices}, Edges: {graph.num_edges}, Colors: {num_colors}")

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color_map[color], markersize=10,
                                    label=f'Color {color}') for color in unique_colors]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {save_path}")

        plt.show()

    def plot_performance_comparison(self, results: Dict[str, Dict],
                                  save_path: Optional[str] = None) -> None:
        """
        Plot performance comparison between different algorithms.

        Args:
            results: Dictionary containing results from different algorithms
            save_path: Path to save the figure (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        algorithms = list(results.keys())
        colors_used = [results[alg]['num_colors'] for alg in algorithms]
        times = [results[alg]['time_taken'] for alg in algorithms]

        # Colors used comparison
        bars1 = ax1.bar(algorithms, colors_used, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Number of Colors Used')
        ax1.set_ylabel('Number of Colors')
        ax1.set_xlabel('Algorithm')

        # Add value labels on bars
        for bar, value in zip(bars1, colors_used):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value}', ha='center', va='bottom')

        # Time comparison (log scale)
        bars2 = ax2.bar(algorithms, times, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Execution Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xlabel('Algorithm')
        ax2.set_yscale('log')

        # Add value labels on bars
        for bar, value in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{value:.6f}', ha='center', va='bottom', rotation=45, fontsize=8)

        # Quality vs Speed scatter plot
        ax3.scatter([times[0]], [colors_used[0]], color='red', s=100, label=algorithms[0])
        for i, alg in enumerate(algorithms[1:], 1):
            ax3.scatter([times[i]], [colors_used[i]], s=100, label=alg)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Number of Colors')
        ax3.set_title('Quality vs Speed Trade-off')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Algorithm efficiency (colors/time ratio)
        efficiency = [c/t if t > 0 else 0 for c, t in zip(colors_used, times)]
        bars4 = ax4.bar(algorithms, efficiency, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax4.set_title('Algorithm Efficiency (Colors/Time)')
        ax4.set_ylabel('Efficiency')
        ax4.set_xlabel('Algorithm')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to {save_path}")

        plt.show()

    def plot_complexity_analysis(self, graph_sizes: List[int],
                                bf_times: List[float],
                                heuristic_times: List[float],
                                save_path: Optional[str] = None) -> None:
        """
        Plot complexity analysis showing how algorithms scale.

        Args:
            graph_sizes: List of graph sizes (number of vertices)
            bf_times: Execution times for brute force
            heuristic_times: Execution times for heuristic
            save_path: Path to save the figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Linear scale comparison
        ax1.plot(graph_sizes, bf_times, 'r-o', label='Brute Force', linewidth=2, markersize=6)
        ax1.plot(graph_sizes, heuristic_times, 'b-s', label='Heuristic', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Vertices')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Algorithm Scaling - Linear Scale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Log scale comparison
        ax2.plot(graph_sizes, bf_times, 'r-o', label='Brute Force', linewidth=2, markersize=6)
        ax2.plot(graph_sizes, heuristic_times, 'b-s', label='Heuristic', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Vertices')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Algorithm Scaling - Log Scale')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Complexity analysis saved to {save_path}")

        plt.show()

    def generate_random_colors(self, n: int) -> List[str]:
        """
        Generate n distinct colors for visualization.

        Args:
            n: Number of colors needed

        Returns:
            List of color strings
        """
        if n <= len(self.colors):
            return self.colors[:n]

        # Generate additional random colors if needed
        additional_colors = []
        for _ in range(n - len(self.colors)):
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            additional_colors.append(color)

        return self.colors + additional_colors

    def save_coloring_to_file(self, graph: Graph, coloring: Dict[int, int],
                            filename: str) -> None:
        """
        Save coloring results to a text file.

        Args:
            graph: The colored graph
            coloring: The coloring dictionary
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(f"Graph Coloring Results\n")
            f.write(f"Vertices: {graph.num_vertices}\n")
            f.write(f"Edges: {graph.num_edges}\n")
            f.write(f"Colors used: {len(set(coloring.values()))}\n")
            f.write(f"\nVertex -> Color mapping:\n")

            for vertex in sorted(coloring.keys()):
                f.write(f"Vertex {vertex}: Color {coloring[vertex]}\n")

            f.write(f"\nEdges:\n")
            for u, v in graph.get_edges():
                f.write(f"({u}, {v}): Color {coloring[u]} - Color {coloring[v]}\n")

        print(f"Coloring results saved to {filename}")