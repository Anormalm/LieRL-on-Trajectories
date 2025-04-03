# Lie Algebra Representation Learning from Trajectories

This project explores a unified deep learning approach to **learning the Lie algebra generators** underlying motion in various **Lie groups** from observed **trajectory sequences**. It provides modular tools to simulate, train, and visualize representation learning on motion data modeled by continuous transformation groups.

Supported groups include:
- **SE(2)**: Planar rigid body motion (2D position + rotation)
- **SE(3)**: Spatial rigid body motion (3D position + rotation)
- **SO(3)**: Pure rotations in 3D
- **SL(2, ℝ)**: Area-preserving linear transformations (non-compact, non-orthogonal)


##  Structure

LieRL/ 
├── src/ 
│ ├── lie_groups.py # Lie group implementations (SE(2), SE(3), SO(3), SL(2,R)) 
│ ├── lieencoder.py # Encoder network definition (MLP) 
│ ├── manifold_plotter.py # Visualization and plotting functions 
│ ├── basis_discovery.py # (Optional) basis and subspace analysis │
├── train/ 
│ ├── train_lie_encoder.py # SE(2) / SE(3) general training pipeline 
│ ├── train_so3_encoder.py # Dedicated SO(3) rotation training 
│ ├── train_sl2r_encoder.py # Dedicated SL(2,R) training with matrix stabilization

---

You can adjust
- `SEQ_LEN`, `DT` (integration timestep)
- Group-specific class in `lie_groups.py`
- Architecture in `lieencoder.py`
---

## Visualizations
Each training script automatically produces:
- **Trajectory overlay**: True vs predicted path
- **Orientation evolution**: Arrows over time (for SO(3))
- **Error curves**: Per-component loss decay
- **Lie algebra bar plots**: True vs predicted generators

---

##  Research Extensions
This codebase supports research in:
- Geometric deep learning on Lie groups
- Representation learning in continuous symmetry spaces
- Error correction under noise and manifold drift
- Learning disentangled bases (see `basis_discovery.py`)
- Applications in robotics, control, and vision

You can easily extend it to other Lie groups (e.g., Sim(2), Heisenberg) by:
1. Defining group operations (`exp`, `log`, `@`, `.inv()`) in `lie_groups.py`
2. Creating a `simulate_*_trajectory()` ...
---


##  Author & License
Developed by [Lifan Hu @Anormalm]. Contributions welcome!

