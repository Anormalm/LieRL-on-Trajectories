import numpy as np

class SE2:
    def __init__(self, R: np.ndarray, t: np.ndarray):
        assert R.shape == (2, 2), "Rotation must be 2x2"
        assert t.shape == (2,), "Translation must be 2D vector"
        self.R = R
        self.t = t

    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        assert mat.shape == (3, 3)
        R = mat[:2, :2]
        t = mat[:2, 2]
        return cls(R, t)

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(3)
        mat[:2, :2] = self.R
        mat[:2, 2] = self.t
        return mat

    def inv(self):
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE2(R_inv, t_inv)

    def __matmul__(self, other):
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return SE2(R_new, t_new)

    def adjoint(self) -> np.ndarray:
        adj = np.eye(3)
        adj[0:2, 0:2] = self.R
        adj[0:2, 2] = self.R @ np.array([[-self.t[1]], [self.t[0]]]).flatten()
        return adj

    @staticmethod
    def hat(xi: np.ndarray) -> np.ndarray:
        vx, vy, theta = xi
        return np.array([
            [0, -theta, vx],
            [theta, 0,  vy],
            [0, 0, 0]
        ])

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        return np.array([mat[0, 2], mat[1, 2], mat[1, 0]])

    @staticmethod
    def exp(xi: np.ndarray) -> 'SE2':
        vx, vy, theta = xi
        if np.isclose(theta, 0.0):
            R = np.eye(2)
            t = np.array([vx, vy])
        else:
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            V = np.array([
                [np.sin(theta)/theta, -(1 - np.cos(theta))/theta],
                [(1 - np.cos(theta))/theta, np.sin(theta)/theta]
            ])
            t = V @ np.array([vx, vy])
        return SE2(R, t)

    def log(self) -> np.ndarray:
        theta = np.arctan2(self.R[1, 0], self.R[0, 0])
        if np.isclose(theta, 0.0):
            vx, vy = self.t
        else:
            A = np.array([
                [np.sin(theta)/theta, -(1 - np.cos(theta))/theta],
                [(1 - np.cos(theta))/theta, np.sin(theta)/theta]
            ])
            V_inv = np.linalg.inv(A)
            vx, vy = V_inv @ self.t
        return np.array([vx, vy, theta])
        
class SE3:
        def __init__(self, R: np.ndarray, t: np.ndarray):
            assert R.shape == (3, 3), "Rotation must be 3x3"
            assert t.shape == (3,), "Translation must be 3D vector"
            self.R = R
            self.t = t

        @classmethod
        def from_matrix(cls, mat: np.ndarray):
            assert mat.shape == (4, 4)
            R = mat[:3, :3]
            t = mat[:3, 3]
            return cls(R, t)

        def as_matrix(self) -> np.ndarray:
            mat = np.eye(4)
            mat[:3, :3] = self.R
            mat[:3, 3] = self.t
            return mat

        def inv(self):
            R_inv = self.R.T
            t_inv = -R_inv @ self.t
            return SE3(R_inv, t_inv)

        def __matmul__(self, other):
            R_new = self.R @ other.R
            t_new = self.R @ other.t + self.t
            return SE3(R_new, t_new)

        def adjoint(self) -> np.ndarray:
            adj = np.zeros((6, 6))
            adj[:3, :3] = self.R
            adj[3:, 3:] = self.R
            skew_t = SE3._skew(self.t)
            adj[3:, :3] = skew_t @ self.R
            return adj

        @staticmethod
        def _skew(v: np.ndarray) -> np.ndarray:
            x, y, z = v
            return np.array([
                [0, -z, y],
                [z,  0, -x],
                [-y, x, 0]
            ])

        @staticmethod
        def hat(xi: np.ndarray) -> np.ndarray:
            w = xi[:3]
            v = xi[3:]
            W = SE3._skew(w)
            mat = np.zeros((4, 4))
            mat[:3, :3] = W
            mat[:3, 3] = v
            return mat

        @staticmethod
        def vee(mat: np.ndarray) -> np.ndarray:
            w = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
            v = mat[:3, 3]
            return np.concatenate([w, v])

        @staticmethod
        def exp(xi: np.ndarray) -> 'SE3':
            w = xi[:3]
            v = xi[3:]
            theta = np.linalg.norm(w)
            if np.isclose(theta, 0.0):
                R = np.eye(3)
                t = v
            else:
                w_unit = w / theta
                W = SE3._skew(w_unit)
                R = (np.eye(3) +
                    np.sin(theta) * W +
                    (1 - np.cos(theta)) * W @ W)
                A = (np.eye(3) +
                    (1 - np.cos(theta)) / theta * W +
                    (theta - np.sin(theta)) / (theta**2) * W @ W)
                t = A @ v
            return SE3(R, t)

        def log(self) -> np.ndarray:
            theta = np.arccos((np.trace(self.R) - 1) / 2)
            if np.isclose(theta, 0.0):
                w = np.zeros(3)
                v = self.t
            else:
                W = (self.R - self.R.T) / (2 * np.sin(theta)) * theta
                w = np.array([W[2, 1], W[0, 2], W[1, 0]])
                W_skew = SE3._skew(w / theta)
                A_inv = (np.eye(3) -
                        0.5 * W_skew +
                        (1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * W_skew @ W_skew)
                v = A_inv @ self.t
            return np.concatenate([w, v])


class SO3:

    def __init__(self, R: np.ndarray):
        assert R.shape == (3, 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), "R must be orthogonal"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "det(R) must be 1"
        self.R = R

    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        return cls(mat)

    def as_matrix(self) -> np.ndarray:
        return self.R

    def inv(self):
        return SO3(self.R.T)

    def __matmul__(self, other):
        return SO3(self.R @ other.R)

    @staticmethod
    def _skew(w: np.ndarray) -> np.ndarray:
        x, y, z = w
        return np.array([
            [0, -z, y],
            [z,  0, -x],
            [-y, x, 0]
        ])

    @staticmethod
    def hat(w: np.ndarray) -> np.ndarray:
        return SO3._skew(w)

    @staticmethod
    def vee(W: np.ndarray) -> np.ndarray:
        return np.array([W[2, 1], W[0, 2], W[1, 0]])

    @staticmethod
    def exp(w: np.ndarray) -> 'SO3':
        theta = np.linalg.norm(w)
        if np.isclose(theta, 0):
            return SO3(np.eye(3))
        w_hat = SO3._skew(w / theta)
        R = (
            np.eye(3) +
            np.sin(theta) * w_hat +
            (1 - np.cos(theta)) * (w_hat @ w_hat)
        )
        return SO3(R)

    def log(self) -> np.ndarray:
        cos_theta = (np.trace(self.R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if np.isclose(theta, 0):
            return np.zeros(3)
        lnR = (theta / (2 * np.sin(theta))) * (self.R - self.R.T)
        return SO3.vee(lnR)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        return self.R @ v
    
    def __init__(self, R: np.ndarray):
        U, _, Vt = np.linalg.svd(R)
        R_orthogonal = U @ Vt
        assert R_orthogonal.shape == (3, 3), "R must be 3x3"
        self.R = R_orthogonal

from scipy.linalg import expm, logm

class SL2R:
    def __init__(self, mat):
        mat = np.asarray(mat, dtype=np.float64)
        det = np.linalg.det(mat)

        if not np.isfinite(det) or det <= 0:
            raise ValueError(f"Invalid SL(2,R) matrix with det={det}")

        scale = det ** (1 / 2)
        mat = mat / scale

        self.mat = mat

    def __matmul__(self, other: 'SL2R') -> 'SL2R':
        return SL2R(self.mat @ other.mat)

    def inv(self) -> 'SL2R':
        return SL2R(np.linalg.inv(self.mat))

    @staticmethod
    def hat(xi: np.ndarray) -> np.ndarray:

        a, b, c = xi
        return np.array([[a, b],
                         [c, -a]])

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:

        return np.array([mat[0, 0], mat[0, 1], mat[1, 0]])

    @staticmethod
    def exp(xi: np.ndarray) -> 'SL2R':
        A = SL2R.hat(xi)
        mat = expm(A)
        det = np.linalg.det(mat)
        if not np.isfinite(det) or abs(det) < 1e-5:
            raise ValueError(f"exp(xi) resulted in invalid matrix with det={det}")
        return SL2R(mat)

    def log(self) -> np.ndarray:
        mat = logm(self.mat)
        return SL2R.vee(mat.real)  

    def as_matrix(self) -> np.ndarray:
        return self.mat

    @staticmethod
    def project_to_SL2R(mat: np.ndarray) -> np.ndarray:
        det = np.linalg.det(mat)
        if not np.isclose(det, 1.0, atol=1e-6):
            mat = mat / np.sqrt(det)
        return mat
