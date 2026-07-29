import numpy as np
import qutip as qt


def apply_wigner(rho: qt.Qobj, grid: int, xvec: float) -> np.ndarray:
    yvec = np.linspace(-xvec, xvec, grid)
    W = qt.wigner(rho, yvec, yvec)
    return np.asarray(W, dtype=np.float32)
