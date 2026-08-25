import numpy as np
import qutip as qt


def apply_husimi(rho: qt.Qobj, grid: int, xvec: float) -> np.ndarray:
    yvec = np.linspace(-xvec, xvec, grid)
    Q = qt.qfunc(rho, yvec, yvec)
    return np.asarray(Q, dtype=np.float32)
