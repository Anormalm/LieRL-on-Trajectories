# Lie Trajectory Encoder 🔁

This project implements deep encoders that learn to predict the underlying Lie algebra generator (`xi`) of simulated trajectories on Lie groups: SE(2), SE(3), SO(3), SL(2, ℝ). Supports noisy trajectory training and robust evaluation.

## 🧪 Supported Groups
- SE(2): Planar motion
- SE(3): 3D rigid body transformations
- SO(3): Pure rotations in 3D
- SL(2, ℝ): Non-compact special linear group

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train SE(2) model
python train_se2_encoder.py
```
