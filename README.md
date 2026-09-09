# Content Moderation Staffing Optimization

This repository implements optimization methods for two-stage stochastic staffing and disposition planning in content moderation systems. The problem features operational constraints (capacity, quality, compliance) and CVaR-based risk management under volume uncertainty. We provide exact decomposition methods (Benders, L-shaped) and scalable heuristics for real-world problem sizes.

## Installation

### Core Dependencies
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

### For Exact Methods
Benders, L-shaped, and Gurobi solvers require:
- Gurobi >= 9.5 with valid license

### For Experiments Only
- openpyxl >= 3.0 (Excel output)

**Note**: The heuristic solver (`Solvers/solver_heuristic.py`) works with SciPy only and does not require Gurobi.

### Install Dependencies
```bash
pip install numpy scipy openpyxl
```

For Gurobi, follow the [official installation guide](https://www.gurobi.com/documentation/).

## Repository Structure

- `Solvers/` - Four solver implementations
  - `solver_gurobi.py` - Direct MIP formulation (deterministic equivalent program)
  - `solver_benders.py` - Benders decomposition with custom cut management
  - `solver_lshaped.py` - L-shaped method with lazy constraint callbacks
  - `solver_heuristic.py` - Scalable heuristic (fix-and-optimize + LNS)

- `ComparativeExperiments/` - Solver comparison experiments from the paper
  - `comparative_experiment.py` - Main solver benchmark
  - `ablation_experiment.py` - Parameter ablation studies
  - `scalability_experiment.py` - Large-scale performance tests

- `SensitivityAnalysisExperiments/` - Parameter sensitivity studies
  - `experiment_m1_anatomy.py` - Cost breakdown analysis
  - `experiment_correlation.py` - Correlation parameter sweep
  - `experiment_joint_value.py` - Risk preference analysis
  - `experiment_vss_and_evpi.py` - Stochastic programming value metrics
  - `experiment_convergence.py` - Scenario count convergence

- `instances/` - Test instances (Small/Medium/Large scales, JSON format)
- `cache/` - Cached experiment results (auto-generated, see `.gitignore`)

**Note**: The experiment scripts are provided for reproducibility. To use the solvers on your own instances, see the Usage section below.

## Usage

### Basic Example

```python
from Solvers import solver_benders
import json

# Load instance
with open("instances/Small_01.json") as f:
    instance = json.load(f)

# Solve with Benders decomposition
result = solver_benders.solve_instance(instance, {
    "time_limit": 3600,
    "threads": 4,
    "output": True
})

# Access solution
print(f"Objective: {result['objective']}")
print(f"Gap: {result['gap']}%")
print(f"Time: {result['time']} seconds")
print(f"Headcount: {result['headcount']}")  # Staffing matrix n[l][k]
print(f"Disposition: {result['disposition']}")  # Disposition matrix y[l][c]
```

### Solver Parameters

All solvers accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_limit` | float | 3600.0 | Time limit in seconds |
| `mip_gap` | float | 0.0 | MIP optimality gap tolerance |
| `threads` | int | 0 | Number of threads (0 = auto-detect) |
| `output` | bool | False | Print solver output to console |

See individual solver files for advanced options (e.g., Benders cut management parameters, heuristic search parameters).

### Running Experiments

To reproduce paper results:

```bash
# Run individual experiment
python -m ComparativeExperiments.comparative_experiment

# Run all experiments (generates all results_*.xlsx files)
python -m Code.RunAllExperiments
```

Cached results are stored in `cache/` (auto-created on first run).

### Instance Format

Instances are JSON files with the following top-level structure:
- `meta`: Instance metadata (name, scale)
- `dimensions`: Problem dimensions (languages, categories, tiers, days, scenarios)
- `sets`: Index sets (tier assignments, action sets, constraint categories)
- `detection_coefficients`: ML detection rates by operating point
- `volume`: Arrival rates and category distributions
- `workload`: Processing times and capacity limits
- `inspection`: Quality control audit rates
- `compliance`: Service level agreements and penalties
- `cost`: Labor costs and content harm costs
- `uncertainty`: Risk parameters (CVaR confidence, risk weight, correlations)
- `scenarios`: Sampled scenarios (probabilities, volume multipliers, capacity realizations)

See `instances/Small_01.json` for a complete example.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
