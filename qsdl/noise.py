import math

import numpy as np
import qutip as qt


def snr_to_sigma(img, snr_db):
    rms = np.sqrt(np.mean(img**2))
    return rms / (10 ** (snr_db / 20))

def gaussian_noise(img, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    return noise + img

def dephase(rho: qt.Qobj, gamma: float) -> qt.Qobj:
    if gamma < 0:
        raise ValueError(f"gamma: {gamma} nie moze byc mniejsza od zera")

    N = rho.shape[0]
    diag = np.arange(N)

    vec = diag[:, None]
    col = diag[None, :]

    decay = np.exp(-gamma/2 * (vec - col) ** 2)

    return qt.Qobj(rho.full() * decay)

def thermal_mix(rho: qt.Qobj, p: float, nbar: float):
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p: {p} musi być między 0 a 1")

    if nbar < 0:
        raise ValueError(f"nbar: {nbar} nie moze byc ujemny")

    N = rho.shape[0]

    rho_thermal = qt.thermal_dm(N, nbar)

    return (1.0 - p) * rho + p * rho_thermal

def photon_loss(rho: qt.Qobj, eta: float) -> qt.Qobj:
    """Pure-loss channel with transmissivity ``eta`` (1 = lossless).

    Uses the exact Kraus decomposition
    ``E_k = sqrt((1-eta)^k / k!) * eta^{n_hat/2} * a^k``.
    """
    if not 0 < eta <= 1:
        raise ValueError("eta must lie in (0, 1]")
    if eta == 1.0:
        return rho
    N = rho.shape[0]
    a = qt.destroy(N)
    eta_half_n = qt.Qobj(np.diag(eta ** (np.arange(N) / 2.0)))
    out = qt.qzero(N)
    a_k = qt.qeye(N)
    for k in range(N):
        if k > 0:
            a_k = a * a_k
        log_coeff = 0.5 * (k * math.log(1 - eta) - math.lgamma(k + 1))
        E_k = math.exp(log_coeff) * (eta_half_n * a_k)
        out += E_k * rho * E_k.dag()
    return out

def generate_params(rng):
    params = {}

    # 30% szum termiczny
    if rng.random() < 0.3:
        params["thermal_p"] = rng.uniform(0.05, 0.3)
        params["thermal_nbar"] = rng.uniform(0.1, 1.0)

    # 70% szans na photon loss
    if rng.random() < 0.7:
        params["photonL"] = rng.uniform(0.6, 0.95)

    # 30% szans na dephase
    if rng.random() < 0.3:
        params["dephase"] = rng.uniform(0.01, 0.2)

    # szum detektora - gaussa 100% szans
    if rng.random() < 1.0:
        params["gauss"] = rng.uniform(10,30)

    return params


def apply_channel_noise(rho: qt.Qobj, params: dict) -> qt.Qobj:
    if "photonL" in params:
        rho = photon_loss(rho, params["photonL"])
    if "dephase" in params:
        rho = dephase(rho, params["dephase"])
    if "thermal_p" in params:
        rho = thermal_mix(rho, params["thermal_p"], params["thermal_nbar"])

    return rho

def apply_wigner_noise(W, params, rng):
    if "gauss" not in params:
        return W

    sigma = snr_to_sigma(W, params["gauss"])
    return gaussian_noise(W, sigma ,rng)
