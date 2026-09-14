from pathlib import Path
from qsdl.dataset import generate_samples
from qsdl.noise import random_level_photon_loss_params

if __name__ == '__main__':
    generate_samples(
        n_per_class=2000,          # 2000*7 = 14000 próbek
        out_path=Path("data/sweep"),
        add_noise=True,
        seed=42,
        noise_fn=random_level_photon_loss_params,
    )
