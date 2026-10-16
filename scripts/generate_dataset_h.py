import argparse
from pathlib import Path

from qsdl.dataset_husimi import generate_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--noise", type=str, nargs="?", const="A", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    noise = args.noise if args.noise is not None else "none"
    if noise not in ["A", "B", "none"]:
        # A = Ahmed noise
        # B = Ours
        raise ValueError(f"Invalid noise type: {noise}")

    generate_samples(args.n_per_class, args.out, noise, args.seed, args.workers)

if __name__ == "__main__":
    main()
