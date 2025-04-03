import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def plot_error_trend(error_log, labels=["v_x", "v_y", "theta"]):
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
    

def plot_se2_trajectory_comparison(true_traj, pred_traj, title="SE(2) Trajectory: Predicted vs True"):
    def extract_xy(traj):
        xs = [p.t[0] for p in traj]
        ys = [p.t[1] for p in traj]
        return xs, ys

    true_x, true_y = extract_xy(true_traj)
    pred_x, pred_y = extract_xy(pred_traj)

    plt.figure(figsize=(6, 6))
    plt.plot(true_x, true_y, 'b-', label='True')
    plt.plot(pred_x, pred_y, 'r--', label='Predicted')

    plt.scatter(true_x[0], true_y[0], c='green', label='Start', s=50)
    plt.scatter(true_x[-1], true_y[-1], c='cyan', label='True End', s=50)
    plt.scatter(pred_x[-1], pred_y[-1], c='magenta', label='Pred End', s=50)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_se3_trajectory_comparison(true_traj, pred_traj, title="SE(3) Trajectory: Predicted vs True"):
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = [], [], []
    for R in poses:
        z_axis = R.R @ np.array([0, 0, 1])  
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

def plot_so3_trajectory_comparison(true_traj, pred_traj, title="SO(3) Orientation Trajectories Comparison"):
    def extract_z_axis(traj):
        return np.array([pose.R @ np.array([0, 0, 1]) for pose in traj])

    true_z = extract_z_axis(true_traj)
    pred_z = extract_z_axis(pred_traj)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_z[:, 0], true_z[:, 1], true_z[:, 2], 'b-', label="True Z-axis")
    ax.plot(pred_z[:, 0], pred_z[:, 1], pred_z[:, 2], 'r--', label="Predicted Z-axis")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sl2r_trajectory_comparison(true_traj, pred_traj, title="SL(2,ℝ) Matrix Action Comparison"):
    v0 = np.array([1.0, 0.0]) 

    true_v = np.array([g.mat @ v0 for g in true_traj])
    pred_v = np.array([g.mat @ v0 for g in pred_traj])

    plt.figure(figsize=(6, 6))
    plt.plot(true_v[:, 0], true_v[:, 1], 'b-', label='True Flow')
    plt.plot(pred_v[:, 0], pred_v[:, 1], 'r--', label='Predicted Flow')

    plt.scatter(true_v[0, 0], true_v[0, 1], c='green', label='Start')
    plt.scatter(true_v[-1, 0], true_v[-1, 1], c='cyan', label='True End')
    plt.scatter(pred_v[-1, 0], pred_v[-1, 1], c='magenta', label='Pred End')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
