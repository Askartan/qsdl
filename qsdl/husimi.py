import numpy as np
import qutip as qt
from scipy.special import gammaln

from qsdl.config import XMAX, CUTOFF


def apply_husimi(rho: qt.Qobj, grid: int, xvec: float) -> np.ndarray:
    yvec = np.linspace(-xvec, xvec, grid)
    Q = qt.qfunc(rho, yvec, yvec)
    return np.asarray(Q, dtype=np.float32)

def apply_husimi_noqfunc(rho: qt.Qobj, C: np.ndarray, grid: int) -> np.ndarray:
    rho_np = np.asarray(rho.full())
    tmp = C.conj() @ rho_np
    Q = np.einsum("gi,gi->g", tmp, C).real / np.pi
    Q = Q.reshape(grid, grid).astype(np.float32)
    return Q

def coherent_amplitudes(
    grid: int, cutoff: int, xmax: float
) -> np.ndarray:
    """
    Matrix ``C[i, n] = <n|alpha_i>`` over the square grid (G, N).

    ``alpha = (x + i p)/sqrt(2)`` (QuTiP ``g = sqrt(2)`` convention);
    computed in log-space so cutoffs up to 64 stay in float range.
    """
    vec = np.linspace(-xmax, xmax, grid)
    alpha = ((vec[None, :] + 1j * vec[:, None]) / np.sqrt(2)).reshape(-1)
    n = np.arange(cutoff)
    # <n|alpha> = e^{-|a|^2/2} a^n / sqrt(n!)
    with np.errstate(divide="ignore"):
        log_a = np.log(np.where(np.abs(alpha) > 0, np.abs(alpha), 1.0))
    log_mag = (-np.abs(alpha[:, None]) ** 2 / 2.0
               + n[None, :] * log_a[:, None]
               - 0.5 * gammaln(n + 1)[None, :])
    phase = np.exp(1j * n[None, :] * np.angle(alpha)[:, None])
    C = np.exp(log_mag) * phase
    C[np.abs(alpha) == 0, 1:] = 0.0  # a^n = 0 for n > 0 at the origin
    C[np.abs(alpha) == 0, 0] = 1.0
    return C.astype(np.complex64)
