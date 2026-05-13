# Graph Coloring Project - CS 301

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-green.svg)](tests/)

A comprehensive implementation and comparison of Graph Coloring algorithms for CS 301 coursework.

## Features

- **Brute Force Algorithm**: Guarantees optimal solutions (exponential time complexity)
- **Heuristic Algorithms**: Four fast approximation algorithms with quality analysis
  - Greedy Coloring
  - Largest Degree First (Welsh-Powell)
  - Smallest Degree Last
  - DSATUR (Degree of Saturation)
- **Performance Analysis**: Comprehensive benchmarking and statistical analysis
- **Visualization**: Graph plotting with coloring results
- **Unit Testing**: Complete test coverage with pytest

## Problem Definition

Given an undirected graph **G(V, E)** and integer **k**, can vertices be colored using **≤ k colors** such that no adjacent vertices share the same color?

**Applications**: Map coloring, scheduling, register allocation, frequency assignment

## Project Structure

```
graph-coloring-project/
├── src/
│   ├── graph.py           # Graph representation and generators
│   ├── brute_force.py     # Optimal O(k^n) algorithm
│   ├── heuristic.py       # Four heuristic algorithms
│   ├── visualization.py   # Plotting and analysis tools
│   └── main.py           # Command-line interface
├── tests/                 # Unit tests (pytest)
├── results/               # Generated plots and analysis
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Installation and Usage

### Installation
```bash
git clone https://github.com/srpcnkr12/graph-coloring-project.git
cd graph-coloring-project
pip install -r requirements.txt
```

### Usage Examples
```bash
# Run demo on sample graphs
python src/main.py --demo

# Performance analysis on graphs up to 12 vertices
python src/main.py --performance --max-vertices 12

# Interactive mode for custom graphs
python src/main.py --interactive

# Run unit tests
pytest tests/ -v
```

## 📊 Algorithm Comparison

| Algorithm | Time Complexity | Space | Quality | Use Case |
|-----------|----------------|-------|---------|----------|
| **Brute Force** | O(k^n) | O(n) | Optimal | Small graphs (n ≤ 8) |
| **Greedy** | O(n·d) | O(n) | ~133% optimal | General purpose |
| **Welsh-Powell** | O(n²) | O(n) | ~100-120% optimal | Dense graphs |
| **DSATUR** | O(n³) | O(n) | ~100-110% optimal | Best quality |

## Sample Results

### Complete Graph K4 (4 vertices, 6 edges)
- **Optimal Colors**: 4 (chromatic number = 4)
- **All Heuristics**: 4 colors (100% optimal)
- **Brute Force Time**: 0.0002 seconds
- **Heuristic Time**: 0.00001 seconds

### Medium Random Graph (8 vertices, 12 edges)
- **Optimal Colors**: 3
- **Greedy**: 4 colors (133% optimal)
- **LDF/SDL/DSATUR**: 3 colors (100% optimal)

## Performance Analysis

For graphs with n vertices:
- **Brute Force**: Practical only for n ≤ 8 vertices
- **Heuristics**: Scale to n > 1000 vertices easily
- **Quality Trade-off**: 0-33% suboptimality for approximately 1000x speedup

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Results**: 9/9 tests passing
- Graph operations
- Algorithm correctness
- Edge cases (empty graphs, single vertex)
- Validation functions

## Future Improvements

- Meta-heuristics: Genetic Algorithm, Simulated Annealing
- Parallelization: Multi-threaded brute force
- Advanced Bounds: Clique-based lower bounds
- Web Interface: Interactive dashboard
- Benchmark Suite: Standard graph datasets (DIMACS)

## Technical Details

**Languages and Libraries:**
- Python 3.8+ with NetworkX, Matplotlib, NumPy
- Testing: pytest framework
- Visualization: matplotlib
- Performance: cProfile profiling support

**Complexity Analysis:**
- Brute force explores all k^n colorings
- Heuristics use greedy vertex ordering strategies
- Space complexity O(V + E) for all algorithms

## Contributors

**CS 301 Spring 2024 Project**
- Implementation and Documentation