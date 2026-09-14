import os
import json
from pathlib import Path

import h5py as hdf
import numpy as np

from qsdl.config import CUTOFF, GRID, XMAX
from qsdl.labels import LABEL_TO_ID, LABELS
from qsdl.noise import apply_channel_noise, apply_gauss_noise, generate_params
from qsdl.states import sample_state
from qsdl.wigner import apply_wigner
from concurrent.futures import ProcessPoolExecutor, as_completed

def _one_sample(args):
    i, label, seed_i, add_noise, noise_fn = args
    rng = np.random.default_rng(seed_i)
    rho, state_data = sample_state(label, CUTOFF, rng)
    rho_np = np.asarray(rho.full(), dtype=np.complex64)

    gen_fn = noise_fn if noise_fn is not None else generate_params
    noise_params = gen_fn(rng) if add_noise else {}
    W_clean = apply_wigner(rho, GRID, XMAX)

    if add_noise:
        rho = apply_channel_noise(rho, noise_params)

    W = apply_wigner(rho, GRID, XMAX)
    if add_noise:
        W = apply_gauss_noise(W, noise_params, rng)

    return (
        i,
        W,
        W_clean,
        rho_np,
        LABEL_TO_ID[label],
        {**state_data, **noise_params}
    )


def generate_samples(n_per_class, out_path, add_noise, seed, workers=None, noise_fn=None):
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
            jobs.append((k, label, seed + k * 10007, add_noise, noise_fn))
            k += 1

    results = [None] * len(jobs)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_sample, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            i, W, W_clean, rho_np, lab, meta = fut.result()
            results[i] = (W, W_clean, rho_np, lab, meta)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    wigners = [r[0] for r in results]
    wigners_clean = [r[1] for r in results]
    rhos_clean = [r[2] for r in results]
    labels = [r[3] for r in results]
    metas = [r[4] for r in results]
    meta_strs = [json.dumps(m, default=float) for m in metas]

    file_name = data_dir / f"train_{"noisy" if add_noise else "clean"}_{n_per_class*7}.h5"

    if file_name.exists() == True:
        file_name = data_dir / f"train_{"noisy" if add_noise else "clean"}_{n_per_class}_copy.h5"

    with hdf.File(f"{file_name}", "w") as f:
        f.create_dataset("wigner", data=wigners)
        f.create_dataset("wigner_clean", data=np.stack(wigners_clean))
        f.create_dataset("rhos_clean", data=np.stack(rhos_clean))
        f.create_dataset("labels", data=labels)
        dt = hdf.string_dtype(encoding="utf-8")
        f.create_dataset("metadata", data=np.array(meta_strs, dtype=object), dtype=dt)

        f.attrs["SEED"] = seed
        f.attrs["noisy"] = add_noise
        f.attrs["LABEL_NAMES"] = list(LABELS)
        f.attrs["CUTOFF"] = CUTOFF
        f.attrs["GRID_SIZE"] = GRID
        f.attrs["XMAX"] = XMAX

    print(f"Zapisano {len(labels)} próbek → {out_path}")
