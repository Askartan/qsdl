import math

import numpy as np
from qutip import Qobj, coherent, coherent_dm, fock, fock_dm, thermal_dm
from scipy.special import binom, gammaln

_GKP_PEAKS = {}


def _gkp_peak_matrix(hilbert_size, mu):
    key = (int(hilbert_size), int(mu))
    cached = _GKP_PEAKS.get(key)
    if cached is not None:
        return cached

    c = np.sqrt(np.pi / 2.0)
    n1 = np.arange(-20, 21)
    n2 = np.arange(-20, 21)
    N1, N2 = np.meshgrid(n1, n2, indexing="ij")
    a = c * (2 * N1 + mu + 1j * N2)
    a = a.reshape(-1)
    n1f = N1.reshape(-1)
    n2f = N2.reshape(-1)
    keep = np.abs(a) ** 2 < hilbert_size * 4
    a = a[keep]
    n1f = n1f[keep]
    n2f = n2f[keep]

    n = np.arange(hilbert_size)
    abs_a = np.abs(a)
    with np.errstate(divide="ignore"):
        log_a = np.log(np.where(abs_a > 0, abs_a, 1.0))
    log_mag = (
        -abs_a[:, None] ** 2 / 2.0
        + n[None, :] * log_a[:, None]
        - 0.5 * gammaln(n + 1)[None, :]
    )
    phase = np.exp(1j * n[None, :] * np.angle(a)[:, None])
    A = np.exp(log_mag) * phase
    zero = abs_a == 0
    A[zero, 1:] = 0.0
    A[zero, 0] = 1.0

    out = (a, n1f, n2f, np.asarray(A, dtype=np.complex128), c)
    _GKP_PEAKS[key] = out
    return out


def fock_state(hilbert_size: int, rng, n=None):
    if n is None:
        n = int(rng.integers(1, 17))

    if not 0 <= n < hilbert_size:
        raise ValueError(f"n={n} poza bazą N={hilbert_size}")

    return fock_dm(hilbert_size, n), {"n": n}


def coherent_state(hilbert_size: int, rng, alpha=None):
    if alpha is None:
        alpha = random_alpha(rng, radius=3, inner_radius=1e-6)

    return coherent_dm(hilbert_size, alpha), {"alpha": alpha}


def vacuum_state(hilbert_size: int, rng):
    return fock_state(hilbert_size, rng, n=0)


def thermal_state(hilbert_size: int, rng, nbar=None):
    if nbar is None:
        nbar = float(rng.uniform(0, 16.0))

    if nbar < 0:
        raise ValueError("nbar >= 0")

    return thermal_dm(hilbert_size, nbar), {"nbar": nbar}


def cat_state(hilbert_size: int, rng, alpha=None, S=None, mu=None):
    """
    Generates a cat state. For a detailed discussion on the definition
    see `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_

    Args:
    -----
        N (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of
                  a computational state for the cat state.


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state density matrix
    """
    if alpha is None:
        alpha = random_alpha(rng, radius=3, inner_radius=1.0)

    if S is None:
        S = int(rng.integers(0, 3))

    if mu is None:
        mu = int(rng.integers(0, 2))

    kend = 2 * S + 1
    cstates = 0 * coherent(hilbert_size, 0)

    for k in range(int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(hilbert_size, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit(), {"alpha": alpha, "S": S, "mu": mu}


def binomial_state(hilbert_size, rng, S=None, N=None, mu=None):
    """
    Binomial code
    """
    if S is None:
        S = int(rng.integers(1, 11))

    if N is None:
        Nmax = (hilbert_size-1) // (S + 1) - 1
        if Nmax > 2:
            N = int(rng.integers(2, Nmax + 1))
        else:
            N = Nmax

    if mu is None:
        mu = int(rng.integers(0, 2))

    c = 1 / math.sqrt(2 ** (N + 1))

    psi = 0 * fock(hilbert_size, 0)

    for m in range(N + 2):
        psi += c * ((-1) ** (mu * m)) * np.sqrt(binom(N + 1, m)) * fock(hilbert_size, (S + 1) * m)

    rho = psi * psi.dag()
    return rho.unit(), {"S": S, "N": N, "mu": mu}


def gkp_state(hilbert_size, rng, delta=None, mu=None):
    """Generates a GKP state
    """
    if mu is None:
        mu = int(rng.integers(0, 2))

    if delta is None:
        delta = float(rng.uniform(0.2, 0.50))

    a, _, _, A, _ = _gkp_peak_matrix(hilbert_size, mu)
    w = np.exp(-delta**2 * np.abs(a) ** 2) * np.exp(-1j * np.real(a) * np.imag(a))
    psi = w @ A
    ket = Qobj(psi.reshape(-1, 1), dims=[[hilbert_size], [1]])
    rho = ket * ket.dag()
    return rho.unit(), {"delta": delta, "mu": mu}


def sample_state(label: str, hilbert_size, rng, **kwargs):
    rng = np.random.default_rng(rng)

    if label == "fock":
        return fock_state(hilbert_size, rng, **kwargs)

    if label == "coherent":
        return coherent_state(hilbert_size, rng, **kwargs)

    if label == "thermal":
        return thermal_state(hilbert_size, rng, **kwargs)

    if label == "vacuum":
        return vacuum_state(hilbert_size, rng)

    if label == "cat":
        return cat_state(hilbert_size, rng, **kwargs)

    if label == "binomial":
        return binomial_state(hilbert_size, rng, **kwargs)

    if label == "gkp":
        return gkp_state(hilbert_size, rng, **kwargs)

    raise ValueError(f"Nieznana klasa: {label!r}")


def random_alpha(rng, radius, inner_radius=0):
    """
    Generates a random complex values within a circle

    Args:
        radius (float): Radius for the values
        inner_radius (float): Inner radius which defaults to 0.
    """
    amp = rng.uniform(inner_radius, radius)
    phi = rng.uniform(-np.pi, np.pi)
    return amp * np.exp(1j * phi)
