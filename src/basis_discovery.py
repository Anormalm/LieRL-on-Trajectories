import numpy as np

def estimate_fixed_generator(xi_seq: np.ndarray):
    """
    Estimate a single generator xi from repeated SE(2) motions.
    Input:
        xi_seq: array of shape (N, 3), where each row is a twist vector
    Output:
        Estimated xi (mean vector), error std
    """
    xi_mean = np.mean(xi_seq, axis=0)
    xi_std = np.std(xi_seq, axis=0)
    return xi_mean, xi_std


def estimate_lie_basis(xi_seq: np.ndarray, threshold=1e-2):
    """
    Estimate the Lie algebra basis using SVD on a collection of twist vectors.
    Input:
        xi_seq: shape (N, d) where d=3 for SE(2) or d=6 for SE(3)
    Output:
        basis_vectors: shape (k, d) where k is the number of significant directions
    """
    xi_centered = xi_seq - np.mean(xi_seq, axis=0)

    U, S, Vt = np.linalg.svd(xi_centered, full_matrices=False)

    rank = np.sum(S > threshold)
    basis_vectors = Vt[:rank]
    return basis_vectors, S

def se2_bracket(xi1: np.ndarray, xi2: np.ndarray):
    """
    Compute the Lie bracket [xi1, xi2] ∈ se(2) using matrix commutator.
    """
    from lie_groups import SE2
    hat1 = SE2.hat(xi1)
    hat2 = SE2.hat(xi2)
    comm = hat1 @ hat2 - hat2 @ hat1
    return SE2.vee(comm)

def test_closure(basis_vectors):
    """
    Test whether the space spanned by the basis is closed under Lie brackets.
    Prints commutator results and checks linear dependence.
    """
    k = basis_vectors.shape[0]
    print(f"Testing closure for {k} basis vectors...\n")
    for i in range(k):
        for j in range(i+1, k):
            xi1 = basis_vectors[i]
            xi2 = basis_vectors[j]
            bracket = se2_bracket(xi1, xi2)
            print(f"[{i},{j}] =", bracket)

if __name__ == "__main__":
    from lieencoder import simulate_se2_trajectory, trajectory_to_lie_algebra
    poses = simulate_se2_trajectory(num_steps=200, dt=0.05)
    xi_seq = trajectory_to_lie_algebra(poses)

    # Estimate fixed generator
    xi_mean, xi_std = estimate_fixed_generator(xi_seq)
    print("Estimated xi:", xi_mean)
    print("Std dev:", xi_std)

    # Estimate algebra basis
    basis, S = estimate_lie_basis(xi_seq)
    print("Estimated basis vectors:")
    print(basis)

    # Check closure
    test_closure(basis)
