## TODO

import argparse
from pathlib import Path

import h5py as hdf
import matplotlib.pyplot as plt
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(f"{Path.cwd().parent}/figures/preview.png"))
    p.add_argument("--n-cols", type=int, default=5) # próbki na klase
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with hdf.File(args.h5, "r") as f:
        rng = np.random.default_rng(args.seed)
        ixs = []
        N = f["wigner"].shape[0]
        n_cols = args.n_cols

        for _ in range(n_cols):
            ix = rng.randint(0, N)
            ixs.append(ix)

        Wigner = [f["wigner"][:][n] for n in ixs]
        Labels = [f["labels"][:][n] for n in ixs]
        Meta = [f["metadata"][:][n] for n in ixs]
        xmax = float(f.attrs.get("XMAX", 6.0))
        n_classes = len(f.attrs.get("LABEL_NAMES"))

    


    fig, axes = plt.subplots(
        n_classes, 
        n_cols,
        figsize=(2.2 * n_cols)
    )

