import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lie_groups import SE2,SE3

class LieEncoder(nn.Module):
    """
    Deep GRU-based encoder for estimating the SE(2) generator
    from a sequence of se(2) twists. Includes dropout, normalization,
    and nonlinearity for better learning.
    """
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.Identity()  # Assume twist components are in range [-1, 1]
        )

    def forward(self, x):
        """
        x: (B, T, 3) twist sequence
        Output: (B, 3) predicted generator
        """
        _, h_n = self.rnn(x)  # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]      # use output of final layer
        h_norm = self.norm(h_last)
        return self.head(h_norm)


def normalize_twist_sequence(x_seq):
    """
    Normalize twist sequence across time dimension (per batch sample).
    Input: x_seq (B, T, 3)
    Output: normalized sequence (B, T, 3), means, stds
    """
    means = x_seq.mean(dim=1, keepdim=True)
    stds = x_seq.std(dim=1, keepdim=True) + 1e-6
    x_norm = (x_seq - means) / stds
    return x_norm, means, stds

def simulate_se2_trajectory(xi, num_steps=100, dt=0.1):
    """
    Simulate a trajectory using a fixed twist xi in SE(2).
    Args:
        xi: 3D twist vector [vx, vy, omega]
        num_steps: number of steps in trajectory
        dt: time step
    Returns:
        list of SE2 poses
    """
    poses = []
    pose = SE2.exp(np.zeros(3))  # identity
    for _ in range(num_steps):
        delta = SE2.exp(xi * dt)
        pose = pose @ delta
        poses.append(pose)
    return poses


def trajectory_to_lie_algebra(poses):
    """
    Convert a sequence of SE2 poses into twist vectors (Lie algebra deltas).
    Args:
        poses: list of SE2 objects
    Returns:
        Numpy array of shape (T-1, 3) of twist vectors
    """
    xi_list = []
    for i in range(1, len(poses)):
        delta = poses[i-1].inv() @ poses[i]
        xi = delta.log()
        xi_list.append(xi)
    return np.array(xi_list)

def simulate_se3_trajectory(true_xi: np.ndarray, seq_len: int, dt: float):
    traj = []
    pose = SE3.exp(np.zeros(6))  # Identity pose
    delta = SE3.exp(true_xi * dt)
    for _ in range(seq_len):
        pose = pose @ delta
        traj.append(pose)
    return traj

