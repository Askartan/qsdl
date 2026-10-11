import math

import numpy as np
import qutip as qt

GAUSS_LEVELS_PCT = tuple(range(0, 101, 5))
# UWAGA: sigma liczona jest wzgledem SZCZYTU |W|, nie calego obrazu (rms calego 128x128
# obrazu jest zdominowane przez tlo bliskie zeru, wiec skala SNR-dB wzgledem rms nigdy
# nie osiagala realnie niszczacego poziomu szumu - patrz uzasadnienie w rozmowie).
GAUSS_SIGMA_FRAC_MAX = 1.5  # przy pct=100 sigma = 150% szczytu |W| -> sygnal calkowicie zatopiony w szumie

def fixed_gauss_params(pct: float) -> dict:
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct={pct} musi być w [0,100]")
    if pct == 0:
        return {"noise_level_pct": 0}          # prawdziwe zero, bez klucza "gauss"
    sigma_frac = (pct / 100.0) * GAUSS_SIGMA_FRAC_MAX
    return {"gauss_peak_frac": sigma_frac, "noise_level_pct": int(round(pct))}

def random_level_gauss_params(rng, levels_pct=GAUSS_LEVELS_PCT):
    pct = rng.choice(levels_pct)
    return fixed_gauss_params(float(pct))


DEPHASE_LEVELS_PCT = tuple(range(0, 101, 5))
GAMMA_MAX = 15.0   # empirycznie: przy CUTOFF=64 to już praktycznie pełna dekoherencja (rho ~diagonalna)

def fixed_dephase_params(pct: float) -> dict:
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct={pct} musi być w [0,100]")
    gamma = (pct / 100.0) * GAMMA_MAX
    return {"dephase": gamma, "noise_level_pct": int(round(pct))}

def random_level_dephase_params(rng, levels_pct=DEPHASE_LEVELS_PCT):
    pct = rng.choice(levels_pct)
    return fixed_dephase_params(float(pct))

PHOTON_LOSS_LEVELS_PCT = tuple(range(0, 101, 5))  # 0,5,...,100

def fixed_photon_loss_params(loss_frac: float) -> dict:
    """loss_frac w [0,1] -> parametry z WYŁĄCZNIE photon lossem, bez gaussa/dephase/thermal."""
    if not 0.0 <= loss_frac <= 1.0:
        raise ValueError(f"loss_frac={loss_frac} musi być w [0, 1]")
    return {"photonL": 1.0 - loss_frac, "noise_level_pct": round(loss_frac * 100)}

def random_level_photon_loss_params(rng, levels_pct=PHOTON_LOSS_LEVELS_PCT):
    """Losuje jeden z dyskretnych poziomów strat i zwraca dla niego parametry."""
    pct = rng.choice(levels_pct)
    return fixed_photon_loss_params(pct / 100.0)

def snr_to_sigma(img, snr_db):
    rms = np.sqrt(np.mean(img**2))
    return rms / (10 ** (snr_db / 20))

def gaussian_noise(img, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    return (noise + img).astype(img.dtype)

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
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0, 1]")
    if eta == 1.0:
        return rho
    if eta == 0.0:
        N = rho.shape[0]
        return qt.fock_dm(N, 0)  # granica pełnej straty = próżnia
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

def apply_gauss_noise(W, params, rng):
    # kontrolowany sweep (fixed_gauss_params) -> sigma jako ulamek szczytu |W|
    if "gauss_peak_frac" in params:
        sigma = params["gauss_peak_frac"] * np.abs(W).max()
        return gaussian_noise(W, sigma, rng)

    # domyslny losowy pipeline (generate_params) -> stara skala SNR w dB, bez zmian
    if "gauss" in params:
        sigma = snr_to_sigma(W, params["gauss"])
        return gaussian_noise(W, sigma, rng)

    return W
