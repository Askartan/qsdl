import h5py as hdf
import numpy as np
import torch
from torch.utils.data import Dataset


class WignerDataset(Dataset):
    def __init__(self, h5_path, indices=None):
        with hdf.File(f"{h5_path}", "r") as f:
            self.wigners: np.ndarray = f["wigner"][:]
            self.labels: np.ndarray = f["labels"][:]
            if indices is not None:
                self.wigners = self.wigners[indices]
                self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        wigner_t = torch.unsqueeze(torch.from_numpy(self.wigners[i]), 0)
        label_t = torch.tensor(self.labels[i], dtype=torch.long)
        return wigner_t, label_t

def stratified_split(N, rng, train, test, val):
    suma = train + test + val
    if not (abs(suma - 1) < 1e-9 or abs(suma - 100)) < 1e-9:
        raise ValueError(f"Parametry train: {train} test: {test} val: {val} musza sie rownac 1 lub 100%")
    if abs(suma - 100) < 1e-9:
        train, test, val = train / 100, test / 100, val / 100

    train_idx, test_idx, val_idx  = [], [], []
    N_class = N // 7

    len_train = int(np.floor(N_class * train))
    len_test = int(np.floor(N_class * test))
    len_val = int(np.floor(N_class * val))

    ixs = np.arange(0, N)

    for i in range(7):
        class_ixs = ixs[i * N_class : (i+1) * N_class]
        shuffled = rng.permutation(class_ixs)
        groups = np.split(shuffled, np.cumsum([len_train, len_test, len_val])[:-1])

        train_idx.extend(list(groups[0]))
        test_idx.extend(list(groups[1]))
        val_idx.extend(list(groups[2]))

    return train_idx, test_idx, val_idx
