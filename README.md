# dynamical_cheeger_visualization

This repository contains the visualization script used to support the CDC manuscript on dynamical Cheeger inequalities for Laplacian networked control systems.

The code computes and visualizes:
- the Riccati spectral map applied to graph Laplacian eigenvalues,
- conductance-based lower and upper bounds induced by Cheeger-type inequalities,
- representative sparse cuts on canonical graph topologies,
- and a comparison between the true Riccati spectral gap and the dynamical Cheeger bound interval.

The script generates two figures:
1. **Representative Graph Topologies and Sparse Cuts**  
   This figure illustrates canonical graph families together with visually identified sparse cuts. Nodes are partitioned into $S$ and $S^c$, internal edges are distinguished from cut edges, and the topology is displayed using fixed layouts chosen for interpretability.

2. **Dynamical Cheeger Bounds vs Riccati Spectral Gap**  
   This figure compares the true value of the Riccati spectral gap against the conductance-based lower and upper bounds predicted by the theory.

## File
- `dynamical_cheeger.py` — main script for computing all quantities and generating both figures.

## Usage

Run the script with Python:

```bash
python dynamical_cheeger.py