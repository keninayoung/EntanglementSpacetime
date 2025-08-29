# Entanglement Spacetime

Author: Kenneth Young, PhD

## Overview
This repository contains code for the project "Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks". 
The framework simulates the emergence of spacetime from quantum entanglement using a time-evolved Projected Entangled Pair States (PEPS) tensor network.
A rigorous 13-stage validation pipeline (R1–R13p) confirms EDG's consistency with GR in weak
fields-including solar-system tests, light bending, Shapiro delay, and Cassini constraints-while
imposing new bounds from Event Horizon Telescope (EHT) ring sizes and
LIGO gravitational-wave (GW) phasing. Evolving 3D entanglement graphs and simulations
reveal dynamic quantum structures potentially resolving black-hole singularities. EDG forecasts
distinct signatures in next-generation EHT and LIGO data, offering a falsifiable path to quantum
gravity.

Core idea:
- Define an information-theoretic distance between lattice sites,
  `d(i,j) ~ -log I(i:j)`, where `I(i:j)` is the mutual information.
- From this, compute discrete curvature, approximate an Einstein-like tensor, and
  analyze holographic entropy and black-hole dynamics.

The code supports:
- Single-GPU execution on Windows and Linux.
- Multi-GPU parallelization on Linux using Dask-CUDA.
- CPU-only mode on both platforms.

For methodology and results, see:

1. [Entanglement-Driven Gravity: Emergent Spacetime from Quantum Correlations and Empirical Constraints](docs/edg-emergent_spacetime.pdf)
2. [Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks: Applications to Quantum and Classical Systems](docs/entanglement-drive-spacetime.pdf)

## EDG Validation Pipeline (R1-R13p)

In addition to simulations, this repository provides a **reproducible validation pipeline** for Entanglement-Driven Gravity (EDG). The stages R1–R13p cover:

- **R1–R6**: Newtonian sanity checks, Bianchi identities, curvature scaling, and numerical convergence of perihelion precession.  
- **R7–R12**: Post-Newtonian observables (light deflection, Shapiro delay, Cassini), PPN cross-checks, bootstrap uncertainty, and integrator consensus.  
- **R13p**: Strong-field joint fit combining EHT photon-ring diameters and GW phasing to constrain \(\epsilon\) and \(L_q\).

The pipeline generates the same CSVs/figures used in the paper (e.g., perihelion tables, PPN tables, ring-posterior plots), and the high-level 

**Validation Flow Diagram**.
![Validation Flow Diagram](docs/figs/validation_flow.png)

## Visual Demos

These visualizations are driven by the quantum outputs written to `spacetime_outputs/`:

- **Quantum-derived orbits** (classical orbits whose central pull is modulated by entropy and Hawking-like mutual information):

  ![Quantum derived orbits](example_outputs/animated_quantum_earth_orbit.gif)

- **Black-hole animation** (stars accrete toward a black hole whose on-plot radius equals a mapped Schwarzschild radius; radius grows smoothly and monotonically based on quantum signals):

  ![Black hole entanglement animation](example_outputs/black_hole_entanglement_2d.gif)

## How the quantum outputs drive the visuals

After you run the PEPS evolution, the following CSVs appear in `spacetime_outputs/`:

- `entropy.csv`: entanglement entropy over time.
- `hawking_radiation.csv`: mutual information across a chosen horizon (Page-curve-like).
- `curvature_lattice.csv` (optional): a 3x3 curvature field per step for simple spatial modulation.

The visualization scripts use these series as follows:

- **Solar system demo (`solar_system_entanglement.py`)**:
  - Global pull `k(t)` is modulated by a normalized blend of `dEntropy/dt` and Hawking MI.
  - Optional 3x3 curvature adds local sector multipliers.
  - Orbits are integrated with Velocity Verlet; this is a classical integrator driven by quantum-derived signals.

- **Black-hole demo (`blackhole_simulation.py`)**:
  - The on-plot black disk radius is set to a mapped Schwarzschild radius `r_s = 2 G M / c^2`.
  - A one-time meter-to-plot mapping is computed using a reference mass and a target on-plot radius.
  - The effective mass is scaled by a smoothed, cumulative, nonnegative driver from the quantum signals.
    This guarantees small, monotonic growth (no shrinking artifacts).

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 12.x for GPU support (optional)
- For larger grids, 32 GB RAM or more is recommended
- Linux for multi-GPU with Dask-CUDA; Windows supports single-GPU or CPU

### Steps
1) Clone the repository:
```bash
git clone https://github.com/yourusername/EntanglementSpacetime.git
cd EntanglementSpacetime
```

2) Create a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

Notes:
- On Linux, install CUDA 12.6 and Dask-CUDA for multi-GPU.
- On Windows, multi-GPU is not supported; the code will use single-GPU or CPU.

## Quick Start - Validations (R1-R13p)

### From the repository root, run:

```bash
python run_validations.py
```
Artifacts produced (paths may vary by --out):

- entanglement_validation/validationoutputs - CSVs used by tables (perihelion, PPN, bootstrap, consensus, posteriors)
- entanglement_validation/validationoutputs - figures (e.g., R4/R6 plots, R13p posterior heatmap, validation flow)

## Usage - Simulation & Visual Demos

### 1) Run the entanglement simulation (writes CSVs used by the demos)

Example 3x3 run:
```bash
python entanglement_spacetime_evolution.py
```

Or via CLI (if you use the CLI wrapper):
```bash
python -m emergent_spacetime.cli --Lx 3 --Ly 3 --steps 5 --hamiltonian heisenberg --use_gpu True
```

**Key args:**
- `--Lx`, `--Ly`: lattice dimensions (default 3x3)
- `--steps`: time steps (default 5)
- `--hamiltonian`: default "heisenberg"
- `--use_gpu`: True or False
- `--approximate`: enable approximate contraction for larger grids

Outputs are written to `spacetime_outputs/`:
- `curvature_evolution.csv`
- `einstein_tensor.csv`
- `entropy.csv`
- `hawking_radiation.csv`
- `curvature_lattice.csv` (if generated by your run)
- `entanglement_graph_tX.html` for each step X

### 2) Generate the solar-system visualization (quantum-driven orbits)
```bash
python solar_system_entanglement.py --use_entanglement --use_spatial_curvature
```

**Useful flags:**
- `--earth_a`, `--mars_a`: tweak semi-major axes without editing code.
- `--trail_len`: length of comet-like trails.
- `--n_steps`, `--dt`: integration and animation time base.

**Artifacts:**
- `spacetime_outputs/animated_quantum_earth_orbit.gif`
- `spacetime_outputs/animated_quantum_earth_orbit.mp4` (if ffmpeg is available)
- `spacetime_outputs/earth_entropy.png`
- `spacetime_outputs/close_Earth_Venus.png`, `close_Earth_Mars.png`

### 3) Generate the black-hole visualization (smooth Schwarzschild growth)
```bash
python blackhole_simulation.py
```

**Behavior:**
- The black disk is sized in data units to match a mapped Schwarzschild radius.
- Growth is monotonic and small; tune in the CONFIG section:
  - `BH_M_REF_SOLAR`, `BH_TARGET_RADIUS_FOR_REF`
  - `BH_GROWTH_MAX` (overall increase cap)
  - `BH_GROWTH_EMA_ALPHA` (smoother or more responsive)
  - weights for the growth driver

**Artifact:**
- `spacetime_outputs/black_hole_entanglement_2d.gif`

## Example Outputs

Example CSVs and HTML are included in `example_outputs/` to make it easy to preview results without running long jobs. For instance:

- **Hawking Radiation (`hawking_radiation.csv`)**: a Page-curve-like MI across the horizon.
- **Curvature Evolution (`curvature_evolution.csv`)**: discrete curvature between site pairs over steps.
- **Entanglement Graphs**: a set of `entanglement_graph_tX.html` files that visualize the evolving MI-weighted graph.

## Project Structure

- `run_validations.py`                   One shot entrypoint for R1-R13p validations.
- `entanglement_validation`              Validation modules, loaders, metrics, are report builders.
- `entanglement_spacetime_evolution.py`  Main PEPS simulation and CSV writers.
- `graph_builder.py`                     Builds the MI graph from PEPS.
- `curvature.py`                         Discrete curvature computation.
- `einstein_tensor.py`                   Einstein-like tensor approximation.
- `entropy.py`                           Entropy helpers.
- `hawking_radiation.py`                 Horizon MI and related metrics.
- `visualization.py`                     3D graph visualizations.
- `solar_system_entanglement.py`         Quantum-driven orbital demo (GIF/MP4).
- `blackhole_simulation.py`              Black-hole demo with mapped Schwarzschild radius.
- `spacetime_outputs/`                   Output directory (created at runtime).
- `example_outputs/`                     Example data and GIFs for README.
- `requirements.txt`                     Dependencies.
- `README.md`                            Project documentation.
- `LICENSE`                              MIT License.

## Reproducing the figures used in the README

1) Run the PEPS simulation to populate `spacetime_outputs/`.
2) Run `solar_system_entanglement.py` and `blackhole_simulation.py`.
3) Copy the generated GIFs into `example_outputs/`:
```bash
cp spacetime_outputs/animated_quantum_earth_orbit.gif example_outputs/
cp spacetime_outputs/black_hole_entanglement_2d.gif example_outputs/
```
4) Commit the updated `example_outputs` and this `README.md`.

## Troubleshooting

- **Black-hole radius looks jumpy or too large**: lower `BH_GROWTH_MAX`, reduce growth weights, or increase `BH_GROWTH_EMA_ALPHA` for heavier smoothing.
- **Orbits too close**: increase `--earth_a` and `--mars_a`, keep safe phase spacing, or lower inner eccentricities. The script prints min Earth-Venus and Earth-Mars separations.
- **ffmpeg not found**: MP4 export is skipped; the GIF is still saved.
- **Validations missing outputs** Validations missing outputs: check --out path permissions and run with -v/--verbose (see --help) to surface stage logs.

## Citation

If you use this code in your research, please cite at least one of the following:  
 
 1. Entanglement-Driven Gravity (EDG) paper
    Kenneth G. Young II, PhD, "Entanglement-Driven Gravity: Emergent Spacetime from Quantum Correlations and Empirical Constraints," 2025.
    [Entanglement-Driven Gravity: Emergent Spacetime from Quantum Correlations and Empirical Constraints](docs/edg-emergent_spacetime.pdf)   
 2. Tensor-network paper (framework foundations)
    Kenneth Young, PhD, "Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks: Applications to Quantum and Classical Systems," 2025.
    [Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks: Applications to Quantum and Classical Systems](docs/entanglement-drive-spacetime.pdf)
 3. Software / repository
    Kenneth G. Young II, PhD, "EntanglementSpacetime: EDG Validation Pipeline and Analysis Code," GitHub, 2025.
    URL: https://github.com/keninayoung/EntanglementSpacetime

## License

MIT License

Copyright (c) 2025 Kenneth Young

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

## Contact

Open an issue or reach out directly to the author.

