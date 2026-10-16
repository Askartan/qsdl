import os
import json
from pathlib import Path
from tqdm import tqdm

import h5py as hdf
import numpy as np

from qsdl.config import CUTOFF, GRID, XMAX
from qsdl.labels import LABEL_TO_ID, LABELS
from qsdl.noise import (
    apply_channel_noise,
    generate_mix_params,
    generate_params,
    mix_random_dm,
)
from qsdl.states import sample_state
from qsdl.husimi import apply_husimi_noqfunc, coherent_amplitudes
from concurrent.futures import ProcessPoolExecutor, as_completed

_C = None


def _json_default(obj):
    if isinstance(obj, (complex, np.complexfloating)):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _init_worker():
    global _C
    _C = coherent_amplitudes(GRID, CUTOFF, XMAX)


def _one_sample(args):
    i, label, seed_i, noise_variant = args
    rng = np.random.default_rng(seed_i)
    rho, state_data = sample_state(label, CUTOFF, rng)
    rho_clean_np = np.asarray(rho.full(), dtype=np.complex64)
    noise_params = {}

    match noise_variant:
        case None | "none":
            pass
        case "A":
            noise_params = generate_mix_params(rng)
            rho = mix_random_dm(
                rho,
                noise_params["mix_sigma"],
                rng,
                noise_params["mix_density"],
            )
        case "B":
            noise_params = generate_params(rng)
            rho = apply_channel_noise(rho, noise_params)
        case _:
            raise ValueError(f"Nieznany noise_variant: {noise_variant!r}")

    husimi = apply_husimi_noqfunc(rho, _C, GRID)
    rho_np = np.asarray(rho.full(), dtype=np.complex64)

    return (
        i,
        husimi,
        rho_clean_np,
        rho_np,
        LABEL_TO_ID[label],
        {**state_data, **noise_params},
    )


def generate_samples(n_per_class, out_path, noise_variant, seed, workers=None):
    if workers is None:
        cpu_count = os.cpu_count() or 1
        workers = max(1, cpu_count - 1)

    project_dir = Path.cwd()
    data_dir = project_dir / f"{out_path}"

    if data_dir.exists() == False:
        data_dir.mkdir(exist_ok=False, parents=True)

    print(f"Zaczynam generowac dane na {os.cpu_count() or 1} procesach")

    jobs = []
    k = 0
    for label in LABELS:
        for _ in range(n_per_class):
            jobs.append((k, label, seed + k * 10007, noise_variant))
            k += 1

    results = [None] * len(jobs)

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = [ex.submit(_one_sample, job) for job in jobs]
        results = [None] * len(jobs)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating husimi samples", unit="sample"):
            i, husimi, rho_clean_np, rho_np, lab, meta = fut.result()
            results[i] = (husimi, rho_clean_np, rho_np, lab, meta)

    husimis = [r[0] for r in results]
    rhos_clean = [r[1] for r in results]
    rhos = [r[2] for r in results]
    labels = [r[3] for r in results]
    metas = [r[4] for r in results]
    meta_strs = [json.dumps(m, default=_json_default) for m in metas]

    tag = "clean" if noise_variant in (None, "none", False) else f"noisy_{noise_variant}"
    file_name = data_dir / f"husimi_train_{tag}_{n_per_class * 7}.h5"

    if file_name.exists() == True:
        file_name = data_dir / f"husimi_train_{tag}_{n_per_class}_copy.h5"

    with hdf.File(f"{file_name}", "w") as f:
        f.create_dataset("husimi", data=husimis)
        f.create_dataset("rhos_clean", data=np.stack(rhos_clean))
        f.create_dataset("rhos", data=np.stack(rhos))
        f.create_dataset("labels", data=labels)

        dt = hdf.string_dtype(encoding="utf-8")
        f.create_dataset("metadata", data=np.array(meta_strs, dtype=object), dtype=dt)

        f.attrs["SEED"] = seed
        f.attrs["noisy"] = noise_variant
        f.attrs["LABEL_NAMES"] = list(LABELS)
        f.attrs["CUTOFF"] = CUTOFF
        f.attrs["GRID_SIZE"] = GRID
        f.attrs["XMAX"] = XMAX

    print(f"Zapisano {len(labels)} próbek → {out_path}")
