import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qsdl.cnn.classifier import StateClassifierCNN
from qsdl.cnn.data import GpuHusimiAugment, HusimiDataset, stratified_split
from qsdl.cnn.evaluation import evaluate
from qsdl.cnn.training import CNN_training
from qsdl.labels import LABELS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GRIDS = (32, 64, 128, 16, 8)


def _h5_path(data_dir: Path, grid: int, n_samples: int) -> Path:
    return data_dir / f"husimi_train_noisy_A_{n_samples}_{grid}.h5"


def _per_class_acc(cm: np.ndarray) -> dict:
    row = cm.sum(axis=1).astype(np.float64)
    row[row == 0] = 1.0
    acc = cm.diagonal() / row
    return {name: float(acc[i]) for i, name in enumerate(LABELS)}


def train_one_grid(grid, args, device):
    h5 = _h5_path(args.data_dir, grid, args.n_samples)
    if not h5.exists():
        raise FileNotFoundError(h5)

    train_idx, test_idx, val_idx = stratified_split(
        args.n_samples, args.split_seed, 80, 10, 10
    )
    use_affine = grid >= 32
    shared = HusimiDataset(h5, augment=False)
    train_ds = HusimiDataset(
        images=shared.images, labels=shared.labels, grid=shared.grid,
        indices=train_idx, augment=False,
    )
    val_ds = HusimiDataset(
        images=shared.images, labels=shared.labels, grid=shared.grid,
        indices=val_idx, augment=False,
    )
    test_ds = HusimiDataset(
        images=shared.images, labels=shared.labels, grid=shared.grid,
        indices=test_idx, augment=False,
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = StateClassifierCNN()
    name = f"husimi_g{grid}"
    history, info = CNN_training(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=args.epochs,
        name=name,
        out_dir=args.out_dir,
        batch_transform=GpuHusimiAugment(affine=use_affine),
    )
    acc, cm = evaluate(model, test_loader, device)
    result = {
        "grid": grid,
        "h5": str(h5),
        "n_samples": args.n_samples,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "affine": use_affine,
        "test_acc": float(acc),
        "per_class_acc": _per_class_acc(np.asarray(cm)),
        "confusion": np.asarray(cm).tolist(),
        "best_val_loss": float(info["best_val_loss"]),
        "best_epoch": int(info["epoch_i"]),
        "checkpoint": info["checkpoint"],
        "history": {k: [float(x) for x in v] for k, v in history.items()},
        "labels": list(LABELS),
    }
    out_json = Path(args.out_dir) / f"{name}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"GRID={grid}  test acc={acc:.4f}  affine={use_affine}  -> {info['checkpoint']}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", type=int, nargs="+", default=list(DEFAULT_GRIDS))
    p.add_argument("--data-dir", type=Path, default=REPO / "data")
    p.add_argument("--out-dir", type=Path, default=REPO / "models")
    p.add_argument("--n-samples", type=int, default=70000)
    p.add_argument("--split-seed", type=int, default=67)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  grids={args.grids}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    summary = []
    for grid in args.grids:
        summary.append(train_one_grid(grid, args, device))

    (args.out_dir / "husimi_grid_summary.json").write_text(
        json.dumps(
            [{"grid": r["grid"], "test_acc": r["test_acc"], "per_class_acc": r["per_class_acc"]} for r in summary],
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
