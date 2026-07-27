import argparse
from pathlib import Path
from qsdl.dataset import generate_samples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--noise", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate_samples(args.n_per_class, args.out, args.noise, args.seed)

if __name__ == "__main__":
    main()