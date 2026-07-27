import qutip as qt
import numpy as np

def apply_wigner(rho: qt.Qobj, grid: int, xvec: float):
    xvec = np.linspace(-xvec, xvec, grid)
    W = qt.wigner(rho, xvec, xvec)   
    return np.asarray(W, dtype=np.float32)