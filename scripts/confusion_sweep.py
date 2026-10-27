"""Macierze pomyłek klasyfikatora (wszystkie 7 stanów) dla każdego etapu sweepa szumu.

Etap = poziom szumu (np. 20%, 25%, ..., 100% strat fotonów). Dla każdego etapu liczone
jest dokładnie to samo co wcześniej przy accuracy (argmax klasyfikatora na zbiorze
testowym), tylko zamiast jednej liczby zapisujemy pełną macierz 7x7.

Potoki (pipelines), które są porównywane:
  noisy          - klasyfikator na zaszumionym Wignerze (wigner)
  reconstructed  - Wigner -> rekonstruktor -> rho_hat -> Wigner(rho_hat) -> klasyfikator
  clean          - klasyfikator na czystym Wignerze (wigner_clean), opcjonalnie

Przykład (z katalogu głównego repo):

    python -m scripts.confusion_sweep     --data scripts/data/sweep/train_noisy_14000.h5     --classifier models/v2_train_noisy_19999.pt     --reconstructor models/sweep_reconstructor.pt     --noise-key photonL     --with-clean     --out results/confusion_photon_loss
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import torch
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from qsdl.cnn.classifier import StateClassifierCNN
from qsdl.cnn.data import stratified_split
from qsdl.config import CUTOFF, GRID, XMAX
from qsdl.labels import LABELS
from qsdl.recon.recon import StateReconstructor
from qsdl.recon.rho import rho_from_params
from qsdl.wigner import apply_wigner

N_CLASSES = len(LABELS)

# jak z parametru w metadanych zrobić "poziom szumu" w % (etap sweepa)
#   photonL: eta w [0,1] (1 = brak strat)  -> straty = (1 - eta) * 100   [zgodne z training_noise_sweep.ipynb]
#   dephase / gauss: zgadnięte wartości domyślne, nadpisz przez --scale / --invert
#   (w repo nie ma skryptów dla sweep_dephase / sweep_gauss, więc sprawdź tabelkę "level_raw_mean" w JSON)
NOISE_KEYS = {
    "photonL": {"scale": 100.0, "invert": True, "name": "Photon loss [%]"},
    "dephase": {"scale": 100.0, "invert": False, "name": "Dephase [%]"},
    "gauss": {"scale": 1.0, "invert": False, "name": "Gauss"},
}


def noise_level(meta: dict, key: str, scale: float, invert: bool) -> int:
    """Poziom szumu próbki; brak parametru w metadanych = próbka bez tego szumu (poziom 0)."""
    if key not in meta:
        return 0
    v = float(meta[key])
    return int(round(scale * ((1.0 - v) if invert else v)))


def load_subset(h5_path, indices, with_clean):
    # h5py wymaga rosnących indeksów; kolejność nie ma znaczenia dla macierzy pomyłek
    idx = np.sort(np.asarray(indices))
    with h5py.File(h5_path, "r") as f:
        wig = f["wigner"][idx].astype(np.float32)
        labels = f["labels"][idx].astype(np.int64)
        meta_raw = f["metadata"][idx]
        wig_clean = f["wigner_clean"][idx].astype(np.float32) if with_clean else None
    metas = [json.loads(m.decode() if isinstance(m, bytes) else m) for m in meta_raw]
    return idx, wig, wig_clean, labels, metas


@torch.no_grad()
def predict(model, wig, device, batch_size):
    model.eval()
    out = []
    for s in range(0, len(wig), batch_size):
        x = torch.from_numpy(wig[s:s + batch_size]).unsqueeze(1).to(device)
        out.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def load_classifier(path, device):
    """Wybiera architekturę po kluczach w checkpoincie: 'net.*' = StateClassifierCNN,
    'features.*'/'head.*' = StateClassifierCNNv2 (importowana z qsdl.cnn.classifier)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = StateClassifierCNN()
    print("Klasyfikator: StateClassifierCNN")
    model.load_state_dict(state)
    return model.to(device)


def _rho_to_wigner(rho_np):
    return apply_wigner(qt.Qobj(rho_np), GRID, XMAX)


@torch.no_grad()
def reconstruct_wigners(recon, wig, device, batch_size, workers):
    """Wigner -> rho_hat (rekonstruktor) -> Wigner(rho_hat) [qutip, równolegle]."""
    recon.eval()
    rhos = []
    for s in tqdm(range(0, len(wig), batch_size), desc="Rekonstrukcja rho"):
        x = torch.from_numpy(wig[s:s + batch_size]).unsqueeze(1).to(device)
        rhos.append(rho_from_params(recon(x), N=CUTOFF).cpu().numpy())
    rhos = np.concatenate(rhos)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        w_rec = list(tqdm(ex.map(_rho_to_wigner, rhos, chunksize=8), total=len(rhos), desc="Wigner z rho_hat"))
    return np.stack(w_rec).astype(np.float32)


def summarize(y_true, y_pred):
    # labels=... gwarantuje zawsze 7x7, nawet jeśli jakiejś klasy brak w danym etapie
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(N_CLASSES), where=support > 0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else None,
        "confusion": cm.tolist(),  # wiersze = prawdziwa klasa, kolumny = predykcja
        "per_class_recall": recall.tolist(),
    }


def plot_cm(cm, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=np.arange(N_CLASSES), yticks=np.arange(N_CLASSES),
        xticklabels=LABELS, yticklabels=LABELS,
        ylabel="Prawdziwa klasa", xlabel="Predykcja", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="plik .h5 ze sweepa (wigner, labels, metadata, ...)")
    p.add_argument("--classifier", type=Path, required=True, help="checkpoint StateClassifierCNN (.pt)")
    p.add_argument("--reconstructor", type=Path, default=None, help="checkpoint StateReconstructor (.pt); bez tego pomijamy potok 'reconstructed'")
    p.add_argument("--noise-key", choices=sorted(NOISE_KEYS), default="photonL", help="który szum definiuje etapy sweepa")
    p.add_argument("--scale", type=float, default=None, help="nadpisz mnożnik parametru -> poziom")
    p.add_argument("--invert", action="store_true", default=None, help="poziom = scale * (1 - parametr)")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test",
                   help="która część danych (podział jak w notebookach: seed 67, 80/10/10)")
    p.add_argument("--split-seed", type=int, default=67)
    p.add_argument("--with-clean", action="store_true", help="dodatkowo potok 'clean' (wigner_clean)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=None, help="procesy do liczenia Wignera z rho_hat")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = NOISE_KEYS[args.noise_key]
    scale = cfg["scale"] if args.scale is None else args.scale
    invert = cfg["invert"] if args.invert is None else args.invert

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with h5py.File(args.data, "r") as f:
        n_total = f["wigner"].shape[0]

    if args.split == "all":
        indices = np.arange(n_total)
    else:
        train_idx, test_idx, val_idx = stratified_split(n_total, args.split_seed, 80, 10, 10)
        indices = {"train": train_idx, "test": test_idx, "val": val_idx}[args.split]
    print(f"Dane: {args.data} | split={args.split} | próbek: {len(indices)}")

    _, wig, wig_clean, labels, metas = load_subset(args.data, indices, args.with_clean)
    raw = np.array([float(m.get(args.noise_key, np.nan)) for m in metas])
    levels = np.array([noise_level(m, args.noise_key, scale, invert) for m in metas])
    level_values = sorted(set(levels.tolist()))
    print(f"Etapy ({cfg['name']}): {level_values}")

    clf = load_classifier(args.classifier, device)

    preds = {"noisy": predict(clf, wig, device, args.batch_size)}

    if args.with_clean:
        preds["clean"] = predict(clf, wig_clean, device, args.batch_size)

    if args.reconstructor is not None:
        rec = StateReconstructor().to(device)
        rec.load_state_dict(torch.load(args.reconstructor, map_location=device, weights_only=True))
        w_rec = reconstruct_wigners(rec, wig, device, args.batch_size, args.workers)
        preds["reconstructed"] = predict(clf, w_rec, device, args.batch_size)

    results = {
        "config": {
            "data": str(args.data), "classifier": str(args.classifier),
            "reconstructor": str(args.reconstructor) if args.reconstructor else None,
            "noise_key": args.noise_key, "scale": scale, "invert": invert,
            "split": args.split, "split_seed": args.split_seed, "labels": list(LABELS),
            "confusion_layout": "wiersze = prawdziwa klasa, kolumny = predykcja",
        },
        "pipelines": {},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, y_pred in preds.items():
        pipe = {"all": summarize(labels, y_pred), "levels": {}}
        (args.out / name).mkdir(exist_ok=True)
        for lvl in level_values:
            m = levels == lvl
            s = summarize(labels[m], y_pred[m])
            s["level_raw_mean"] = float(np.nanmean(raw[m])) if np.any(~np.isnan(raw[m])) else None
            pipe["levels"][str(lvl)] = s
            plot_cm(s["confusion"], f"{name} | {cfg['name']} = {lvl} | acc = {s['accuracy']:.3f} (n={s['n']})",
                    args.out / name / f"level_{lvl:03d}.png")
        results["pipelines"][name] = pipe
        print(f"[{name}] accuracy całość: {pipe['all']['accuracy']:.4f}")

    with open(args.out / "confusion.json", "w") as f:
        json.dump(results, f, indent=2)

    # podsumowanie: accuracy vs etap
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, pipe in results["pipelines"].items():
        ax.plot(level_values, [pipe["levels"][str(l)]["accuracy"] for l in level_values], marker="o", label=name)
    ax.set(xlabel=cfg["name"], ylabel="Accuracy klasyfikatora", ylim=(0, 1.02), title="Accuracy vs etap sweepa")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.out / "accuracy_vs_level.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Zapisano → {args.out}")


if __name__ == "__main__":
    main()
