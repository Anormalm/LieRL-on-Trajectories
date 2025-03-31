import numpy as np
import matplotlib.pyplot as plt
from lie_groups import SE2


def plot_se2_trajectory(poses, title="SE(2) Trajectory", show_orientations=True):
    """
    Plot a 2D trajectory generated in SE(2).
    Args:
        poses: list of SE2 objects
        title: plot title
        show_orientations: if True, draw arrows for heading direction
    """
    xs = [pose.t[0] for pose in poses]
    ys = [pose.t[1] for pose in poses]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, label="Trajectory", linewidth=2)
    plt.scatter(xs[0], ys[0], c='green', label='Start')
    plt.scatter(xs[-1], ys[-1], c='red', label='End')

    if show_orientations:
        for i in range(0, len(poses), max(1, len(poses)//30)):
            direction = poses[i].R @ np.array([0.2, 0.0])
            plt.arrow(xs[i], ys[i], direction[0], direction[1], head_width=0.05, color='blue')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lie_error(true_xi, pred_xi, title="Generator Estimation Error"):
    """
    Plot bar chart of estimation error for xi components.
    Handles both SE(2) and SE(3) cases.
    """
    true_xi = np.array(true_xi)
    pred_xi = np.array(pred_xi)
    err = pred_xi - true_xi

    dim = len(true_xi)
    if dim == 3:
        labels = ["v_x", "v_y", "theta"]
    elif dim == 6:
        labels = ["w_x", "w_y", "w_z", "v_x", "v_y", "v_z"]
    else:
        labels = [f"xi_{i}" for i in range(dim)]

    x = np.arange(dim)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, true_xi, width=0.4, label='True')
    plt.bar(x + 0.2, pred_xi, width=0.4, label='Predicted')
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("Value")
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Absolute Errors:", np.abs(err))



def plot_error_trend(error_log, labels=["v_x", "v_y", "theta"]):
    """
    Plot training error per component over time.
    Args:
        error_log: list of (vx_err, vy_err, th_err) tuples
    """
    error_log = np.array(error_log)
    plt.figure(figsize=(6, 4))
    for i in range(error_log.shape[1]):
        plt.plot(error_log[:, i], label=f"Error: {labels[i]}")
    plt.title("Per-Component Absolute Error During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def integrate_trajectory(xi: np.ndarray, seq_len: int, dt: float):
    pose = SE2.exp(np.zeros(3))  # Start at identity
    traj = []

    for _ in range(seq_len):
        delta = SE2.exp(xi * dt)
        pose = pose @ delta
        traj.append(pose)

    return traj


def plot_predicted_vs_true_trajectory(true_xi, pred_xi, seq_len, dt):
    true_traj = integrate_trajectory(true_xi, seq_len, dt)
    pred_traj = integrate_trajectory(pred_xi, seq_len, dt)

    true_xy = np.array([[p.t[0], p.t[1]] for p in true_traj])
    pred_xy = np.array([[p.t[0], p.t[1]] for p in pred_traj])

    plt.figure(figsize=(6, 6))
    plt.plot(true_xy[:, 0], true_xy[:, 1], 'b-', label='True')
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'r--', label='Predicted')
    plt.scatter(true_xy[0, 0], true_xy[0, 1], c='blue', label='Start')
    plt.legend()
    plt.axis('equal')
    plt.title("Trajectory: Predicted vs True (SE(2))")
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax 
import numpy as np

def plot_se3_trajectory_comparison(true_traj, pred_traj, title="SE(3) Trajectory: Predicted vs True"):
    """
    Plot 3D SE(3) trajectories of predicted vs true.
    Args:
        true_traj: list of SE3 objects (ground truth)
        pred_traj: list of SE3 objects (predicted)
    """
    def extract_xyz(traj):
        xs = [p.t[0] for p in traj]
        ys = [p.t[1] for p in traj]
        zs = [p.t[2] for p in traj]
        return xs, ys, zs

    true_x, true_y, true_z = extract_xyz(true_traj)
    pred_x, pred_y, pred_z = extract_xyz(pred_traj)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_x, true_y, true_z, 'b-', label='True')
    ax.plot(pred_x, pred_y, pred_z, 'r--', label='Predicted')

    ax.scatter(true_x[0], true_y[0], true_z[0], c='green', label='Start', s=50)
    ax.scatter(true_x[-1], true_y[-1], true_z[-1], c='cyan', label='True End', s=50)
    ax.scatter(pred_x[-1], pred_y[-1], pred_z[-1], c='magenta', label='Pred End', s=50)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_so3_orientation_evolution(poses, title="SO(3) Orientation Evolution"):
    """
    Plot the evolution of Z-axis (forward direction) of SO(3) rotations in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = [], [], []
    for R in poses:
        z_axis = R.R @ np.array([0, 0, 1])  # <-- FIXED: use .R to access rotation matrix
        xs.append(z_axis[0])
        ys.append(z_axis[1])
        zs.append(z_axis[2])

    ax.plot(xs, ys, zs, color='blue', label="Z-axis tip trajectory")
    ax.scatter(xs[0], ys[0], zs[0], c='green', label="Start", s=50)
    ax.scatter(xs[-1], ys[-1], zs[-1], c='red', label="End", s=50)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


