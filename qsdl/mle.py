import time
import numpy as np
from scipy.special import gammaln


def mle_husimi(Qmap, cutoff, xmax, max_iters):

    grid = Qmap.shape[0]
    f = np.clip(Qmap.reshape(-1), 0.0, None)
    f = f / f.sum()

    C = coherent_amplitudes(grid, cutoff, xmax).astype(np.complex128)
    G = C.T @ C.conj()

    G_eigval, G_eigvec = np.linalg.eigh(G)
    g_inv_sqrt = (G_eigvec / np.sqrt(np.clip(G_eigval, 1e-12, None))) @ G_eigvec.conj().T
    rho = G / np.trace(G).real

    t0 = time.perf_counter()
    for it in range(max_iters):
        p = np.einsum("gn,nm,gm->g", C.conj(), rho, C).real
        ratio = np.where(p > 1e-300, f / np.clip(p, 1e-300, None), 0.0)
        R = (C.T * ratio) @ C.conj()
        T = g_inv_sqrt @ R
        new = T @ rho @ T.conj().T
        new = (new + new.conj().T) / 2.0
        new /= np.trace(new).real
        if np.abs(new - rho).sum() < 1e-10:
            rho = new
            break
        rho = new
    return {"rho": rho, "iters": it + 1, "time_s": time.perf_counter() - t0}


#TODO napisac pozniej po swojemu
def coherent_amplitudes(
    grid: int, cutoff: int, xmax: float = 5.0
) -> np.ndarray:
    """Matrix ``C[i, n] = <n|alpha_i>`` over the square grid (G, N).

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
