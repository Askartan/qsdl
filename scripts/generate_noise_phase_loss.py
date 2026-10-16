from pathlib import Path
from qsdl.dataset import generate_samples
from qsdl.noise import random_level_dephase_params

if __name__ == '__main__':
    generate_samples(
        n_per_class=2000,
        out_path=Path("data/sweep_dephase"),
        noise_variant="B",
        seed=42,
        noise_fn=random_level_dephase_params,
    )
