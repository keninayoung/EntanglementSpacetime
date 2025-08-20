"""
Command-line interface for the EntanglementSpacetime Simulation Framework.

This script allows running entanglement-driven spacetime simulations from the command line,
specifying lattice dimensions, bond dimension, time steps, and other parameters.

Usage Examples:
--------------
# Run 3x3 lattice simulation
python cli.py --Lx 3 --Ly 3 --bond_dim 4 --time_steps 5 --dt 0.1 --use_gpu True --approximate True --output_dir spacetime_outputs_3x3_original_2025_05_22

# Run 4x4 lattice simulation
python cli.py --Lx 4 --Ly 4 --bond_dim 4 --time_steps 20 --dt 0.05 --use_gpu True --approximate True --diagonal_bipartition True --pre_gate_compression True --bipartition_type diagonal --output_dir spacetime_outputs_4x4_2025_05_22

# Run 5x5 lattice simulation
python cli.py --Lx 5 --Ly 5 --bond_dim 4 --time_steps 20 --dt 0.05 --use_gpu True --approximate True --diagonal_bipartition True --pre_gate_compression True --bipartition_type diagonal --output_dir spacetime_outputs_5x5_2025_05_22

Requirements:
-------------
- Python 3.9+
- Dependencies: numpy, quimb, cupy, networkx, pandas, cotengra, pyvis, tqdm, psutil, scipy, dask, scikit-learn
- Install via: conda install -c conda-forge numpy=1.25.2 quimb=1.11.0 cupy networkx pandas cotengra tqdm pyvis psutil scipy dask scikit-learn
- For multi-GPU (Linux): conda install -c conda-forge dask-cuda
"""
import time
import argparse
from entanglement_spacetime_evolution import run_simulation

def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="Entanglement Spacetime Evolution")
    parser.add_argument("--Lx", type=int, default=3, help="Lattice width")
    parser.add_argument("--Ly", type=int, default=3, help="Lattice height")
    parser.add_argument("--bond_dim", type=int, default=2, help="Bond dimension")
    parser.add_argument("--time_steps", type=int, default=5, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")
    parser.add_argument("--J", type=float, default=1.0, help="Heisenberg coupling")
    parser.add_argument("--approximate", action="store_true", help="Use approximate MI")
    parser.add_argument("--output_dir", type=str, default="spacetime_outputs", help="Output directory")
    
    args = parser.parse_args()
    
    start_time = time.time()
    run_simulation(
        Lx=args.Lx,
        Ly=args.Ly,
        bond_dim=args.bond_dim,
        time_steps=args.time_steps,
        dt=args.dt,
        J=args.J,
        approximate=args.approximate,
        output_dir=args.output_dir,
    )
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()