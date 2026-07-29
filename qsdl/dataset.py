import json
from pathlib import Path

import h5py as hdf
import numpy as np

from qsdl.config import CUTOFF, GRID, XMAX
from qsdl.labels import LABEL_TO_ID, LABELS
from qsdl.noise import apply_channel_noise, apply_wigner_noise, generate_params
from qsdl.states import sample_state
from qsdl.wigner import apply_wigner


def generate_samples(n_per_class: int, out_path, add_noise: bool, seed: int = 67):
    rng = np.random.default_rng(seed)
    wigners, labels, metas = [], [], []

    project_dir = Path.cwd()
    data_dir = project_dir / f"{out_path}"

    if data_dir.exists() == False:
        data_dir.mkdir(exist_ok=False, parents=True)

    for label in LABELS:
        print(f"Generuję {n_per_class} probek stanu {label}: {LABEL_TO_ID[label] + 1} / 7 ... ", end="", flush=True,)

        for _ in range(n_per_class):
            rho, state_data = sample_state(label, CUTOFF, rng)

            noise_params = generate_params(rng) if add_noise else {}

            if add_noise:
                rho = apply_channel_noise(rho, noise_params)

            W = apply_wigner(rho, GRID, XMAX)

            if add_noise:
                W = apply_wigner_noise(W, noise_params, rng)

            wigners.append(W)
            labels.append(LABEL_TO_ID[label])
            metas.append({**state_data, **noise_params})

        print("OK")

    Wigner_array = np.stack(wigners).astype(np.float32)
    Label_array = np.array(labels, dtype=np.int8)

    Meta_strs = [json.dumps(m, default=float) for m in metas]

    file_name = data_dir / f"train_{"noisy" if add_noise else "clean"}_{n_per_class*7}.h5"

    if file_name.exists() == True:
        file_name = data_dir / f"train_{"noisy" if add_noise else "clean"}_{n_per_class}_copy.h5"

    with hdf.File(f"{file_name}", "w") as f:
        f.create_dataset("wigner", data=Wigner_array)
        f.create_dataset("labels", data=Label_array)
        dt = hdf.string_dtype(encoding="utf-8")
        f.create_dataset("metadata", data=np.array(Meta_strs, dtype=object), dtype=dt)

        f.attrs["SEED"] = seed
        f.attrs["noisy"] = add_noise
        f.attrs["LABEL_NAMES"] = list(LABELS)
        f.attrs["CUTOFF"] = CUTOFF
        f.attrs["GRID_SIZE"] = GRID
        f.attrs["XMAX"] = XMAX

    print(f"Zapisano {len(labels)} próbek → {out_path}")
