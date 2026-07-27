import qutip as qt
from qutip import fock_dm, coherent_dm, thermal_dm 

import math
import numpy as np
from config import CUTOFF

# stan |n><n|
def fock_state(N: int, n: int) -> qt.Qobj:
    if not 0 <= n < N:
        raise ValueError(f"n={n} poza bazą N={N}")
    
    return fock_dm(N, n) 

def coherent_state(N: int, alpha: complex) -> qt.Qobj:
    if 0 >= N:
        raise ValueError(f"baza N={N} musi byc wieksza od 0")
    
    return coherent_dm(N, alpha)

# stan |0><0|
def vacuum_state(N: int) -> qt.Qobj:
    return fock_state(N, 0)

def thermal_state(N: int, nbar: float) -> qt.Qobj:
    if nbar < 0:
        raise ValueError("nbar >= 0")

    return thermal_dm(N, nbar)


def cat_state(N: int, alpha: complex, phi: float) -> qt.Qobj:
    ket = qt.coherent(N, alpha) + np.exp(1j * phi) * qt.coherent(N, -alpha)
    return qt.ket2dm(ket.unit())


def binomial_state(N: int, S: int, K: int, mu: int = 0) -> qt.Qobj:
    """Binomial-code word (Michael et al., PRX 6, 031006 (2016)).

    ``|W_mu> ~ sum_{p = mu mod 2} sqrt(C(K+1, p)) |p (S+1)>`` for
    ``p = 0 .. K+1``.  ``S`` sets the Fock-space spacing (protects against
    up to ``S`` photon losses), ``K`` the code order, ``mu`` in {0, 1}
    selects the logical word.
    """
    if mu not in (0, 1):
        raise ValueError("mu must be 0 or 1")
    max_n = (K + 1) * (S + 1)
    if max_n >= N:
        raise ValueError(
            f"binomial code needs {max_n + 1} Fock levels, cutoff is {N}"
        )
    ket = qt.Qobj(np.zeros((N, 1)))
    for p in range(mu, K + 2, 2):
        ket += math.sqrt(math.comb(K + 1, p)) * qt.basis(N, p * (S + 1))
    return qt.ket2dm(ket.unit())

def gkp_state(N: int, delta: float, mu: int = 0, smax: int = 4) -> qt.Qobj:
    """Finite-energy square-lattice GKP state.

    Built as a comb of position-squeezed states displaced along ``x`` by
    ``(2s + mu) * sqrt(pi/2)`` for ``s = -smax .. smax``, followed by the
    Gaussian envelope operator ``exp(-delta^2 * n_hat)``.  ``delta``
    controls both peak width and envelope (typical physical values
    0.2-0.5); ``mu`` in {0, 1} selects the logical word.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")

    if mu not in (0, 1):
        raise ValueError("mu must be 0 or 1")

    r = -math.log(delta)  # squeezing so that peak width ~ delta
    peak = qt.squeeze(N, r) * qt.basis(N, 0)
    ket = qt.Qobj(np.zeros((N, 1)))

    for s in range(-smax, smax + 1):
        alpha = math.sqrt(math.pi / 2.0) * (2 * s + mu)
        ket += qt.displace(N, alpha) * peak

    envelope = qt.Qobj(np.diag(np.exp(-(delta**2) * np.arange(N))))
    ket = envelope * ket
    norm = ket.norm()

    if norm < 1e-6:
        raise ValueError("GKP state vanished after truncation; increase N")
   
    return qt.ket2dm(ket / norm)

def sample_state(label: str, N: int, rng) -> tuple[qt.Qobj, dict]:
    if label == "fock":
        n = int(rng.integers(1,9))
        
        return fock_state(CUTOFF, n), {"n": n}

    if label == "coherent":
        amp = float(rng.uniform(0.5 , 4.0))
        phase = rng.uniform(0, 2*np.pi)
        alpha = amp * np.exp(1j *phase)

        return coherent_state(CUTOFF, alpha), {"amp": amp, "phase": phase, "alpha_re": alpha.real, "alpha_im": alpha.imag}

    if label == "thermal":
        nbar = float(rng.uniform(0.2, 4.0))
        
        return thermal_state(CUTOFF, nbar), {"nbar": nbar}

    if label == "vacuum":
        return vacuum_state(CUTOFF), {"n": 0}

    if label == "cat":
        amp = float(rng.uniform(1.0 , 3.0))
        phase = rng.uniform(0, 2*np.pi)
        alpha = amp * np.exp(1j *phase)

        phi = float(rng.choice([0.0, np.pi]))

        return cat_state(CUTOFF, alpha, phi), {"amp": amp, "phase": phase, "alpha_re": alpha.real, "alpha_im": alpha.imag, "phi": phi}

    if label == "binomial":
        S = int(rng.integers(1, 3))      
        K = int(rng.integers(2, 5))      
        mu = int(rng.integers(0, 2))     
        return binomial_state(CUTOFF, S, K, mu), {"S": S, "K": K, "mu": mu}

    if label == "gkp":
        delta = float(rng.uniform(0.25, 0.45))
        mu = int(rng.integers(0, 2))
        return gkp_state(CUTOFF, delta, mu), {"delta": delta, "mu": mu, "smax": 4}

    raise ValueError(f"Nieznana klasa: {label!r}")
