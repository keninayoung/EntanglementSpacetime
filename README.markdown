# EntanglementSpacetime

## Overview

This repository contains the code for the project "Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks" by Kenneth Young, PhD. The framework simulates the emergence of spacetime from quantum entanglement using a time-evolved Projected Entangled Pair States (PEPS) tensor network. It defines spacetime distances as \(d(i, j) \sim -\log I(i:j)\), where \(I(i:j)\) is the mutual information between quantum sites, computes discrete curvature, approximates the Einstein tensor, and analyzes holographic entropy and black hole dynamics.

The code supports:
- Single-GPU execution on Windows and Linux.
- Multi-GPU parallelization on Linux using Dask-CUDA.
- CPU parallelization on both platforms.

For methodology and results, see the publication:  
"Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks" by Kenneth Young, PhD (2025).

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 12.x (for GPU support)
- Minimum 16 GB RAM for 4x4 grid simulations
- Linux (for multi-GPU parallelization; Windows supports single-GPU or CPU)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EntanglementSpacetime.git
   cd EntanglementSpacetime
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: For multi-GPU support on Linux, ensure CUDA 12.6 and Dask-CUDA are installed. On Windows, multi-GPU is not supported; the code will fall back to single-GPU or CPU.

## Usage

Run a simulation using the command-line interface. Example for a 3x3 grid with GPU support:
```bash
python3 -m emergent_spacetime.cli --Lx 3 --Ly 3 --steps 5 --hamiltonian heisenberg --use_gpu True
```

### Command-Line Arguments
- `--Lx`: Lattice width (default: 3)
- `--Ly`: Lattice height (default: 3)
- `--steps`: Number of time steps (default: 5)
- `--hamiltonian`: Hamiltonian type (default: "heisenberg")
- `--use_gpu`: Enable GPU acceleration (default: True)
- `--approximate`: Use approximate contraction for large grids (default: False)

## Outputs

Results are saved in the `spacetime_outputs` directory:
- `curvature_evolution.csv`: Discrete curvature over time steps.
- `einstein_tensor.csv`: Einstein tensor approximation.
- `entropy.csv`: Entanglement entropy over time.
- `hawking_radiation.csv`: Mutual information across the horizon (Page curve).
- `entanglement_graph_tX.html`: Interactive 3D visualizations for each time step \(t\).

### Example Outputs
Example output files from a 3x3 simulation (as reported in the paper) are provided in the `example_outputs` directory to help users understand the simulation results:

- **Curvature Evolution (`curvature_evolution.csv`)**: Discrete curvature \( \kappa(i,j) \) between pairs of sites over 5 time steps.
  ```
  ,Step 0,Step 1,Step 2,Step 3,Step 4
  0-1,-0.089448,...,...,...,...
  0-2,...,...,...,...,...
  ```
  Reported in the paper: \( \kappa(0,1) = -0.089448 \) at step 0, indicating AdS-like negative curvature.

- **Hawking Radiation (`hawking_radiation.csv`)**: Mutual information across the horizon (middle row of the lattice), resembling a Page curve.
  ```
  Step,MI Across Horizon
  0,0.39
  1,...
  2,0.63
  3,...
  4,...
  ```
  Reported in the paper: MI varies from 0.39 to 0.63, peaking at 0.63 in step 2, indicating unitary evolution.

- **Entanglement Graph (`entanglement_graph_t2.html`)**: Interactive 3D visualization of the entanglement graph at time step 2 (where the Page curve peaks). Nodes represent quantum sites, and edges are weighted by mutual information \( I(i:j) \), with \( d(i,j) \sim -\log I(i:j) \). Open this file in a web browser to explore the emergent spacetime geometry. An example of the entanglement graph at time step 4 is below:

![entanglement_graph_t4_img](https://github.com/user-attachments/assets/7323cafa-2c46-40f6-9bad-8d28c20ed0d0)

Users can run the simulation themselves to generate the full set of outputs for different parameters (e.g., 4x4 grid).

## Project Structure
- `emergent_spacetime_evolution.py`: Main simulation script.
- `graph_builder.py`: Builds the entanglement graph.
- `curvature.py`: Computes discrete curvature.
- `einstein_tensor.py`: Approximates the Einstein tensor.
- `entropy.py`: Computes entanglement entropy.
- `hawking_radiation.py`: Analyzes black hole dynamics.
- `visualization.py`: Generates interactive visualizations.
- `spacetime_outputs/`: Directory for simulation outputs (empty in repository).
- `example_outputs/`: Example output files from a 3x3 simulation.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.
- `LICENSE`: MIT License.

## Citation
If you use this code in your research, please cite:  
Kenneth Young, PhD "Entanglement-Driven Emergent Spacetime with Time-Evolved Tensor Networks: Applications to Quantum and Classical Systems," 2025.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, please open an issue on GitHub or contact the author directly.
