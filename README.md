# Genetic Algorithm for the Traveling Salesman Problem (TSP)

This project implements a **Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP)**. It includes features such as **local search optimization, clustering-based hierarchical GA**, and parallel execution to improve efficiency.

## Features

- 🧬 **Genetic Algorithm (GA)**: Evolutionary approach to find near-optimal TSP solutions.
- 🔍 **Optional Local Search**: Enhance solutions by applying local optimizations, such as: 2-opt, Hill Climbing and combined local search operators
- 🏙️ **Clustering (Optional)**: Divide cities into clusters and apply a two-level GA:
  - **Level 1**: Solve TSP within each cluster.
  - **Level 2**: Solve TSP at the inter-cluster level.
- ⚡ **Parallel Execution**: Speed up computations using multiprocessing.
- 📊 **Plots & Visualization**: View route optimizations and performance metrics, such as fitness, population diversity over generations, and computational time for each algorithm step for each generation.

## Upcoming Features 🚀

💻 **Online Dashboard (Work in Progress!)**  
- Run simulations directly from a web interface.  
- Choose different configurations: **apply local search, enable clustering, adjust parameters**.  
- Get **interactive plots** and detailed simulation results.  

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/TSP_GA.git
cd TSP_GA
```

Installing the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run a basic TSP simulation using GA:

```
from algorithm import algorithm

# Initialize the algorithm with a given number of cities
tsp_solver = algorithm(num_cities=50)

# Run the GA with local search enabled, clustering disabled
tsp_solver.run_algorithm_main(generateDataSets=True, clusters=False, local_search=True)
```

Run with Clustering (Hierarchical GA):
```
# Run the GA with clustering enabled (two-level GA)
tsp_solver.run_algorithm_main(generateDataSets=True, clusters=True, local_search=True)
```

## Results & Visualization
- The algorithm provides distance matrices, best route solutions, and fitness scores.
- Plots: Execution times, fitness evolution, and optimal routes.
- Results will soon be available online via a dashboard interface using Flask.


## Future Improvements
## Results & Visualization
- 🌍 Web Dashboard Integration
- 🎯 More Advanced Local Search Techniques
- 🚀 Multi-Objective Optimization (Speed vs. Distance Trade-offs)
- ⚡ Further Parallelization & Performance Boosts

