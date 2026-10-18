import h5py as hdf
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip


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
        wigner_t = torch.unsqueeze(torch.from_numpy(self.wigners[i]).float(), 0)
        label_t = torch.tensor(self.labels[i], dtype=torch.long)
        return wigner_t, label_t


class HusimiDataset(Dataset):
    def __init__(self, h5_path, indices=None, augment=False, affine=True, gauss_sigma_max=0.05):
        with hdf.File(f"{h5_path}", "r") as f:
            self.images = np.asarray(f["husimi"][:], dtype=np.float32)
            self.labels = np.asarray(f["labels"][:])
            self.grid = int(f.attrs.get("GRID_SIZE", self.images.shape[-1]))
            if indices is not None:
                idx = np.asarray(indices)
                self.images = self.images[idx]
                self.labels = self.labels[idx]

        self.augment = bool(augment)
        self.affine = bool(affine) and self.grid >= 32
        self.gauss_sigma_max = float(gauss_sigma_max)
        self._affine = None
        if self.augment and self.affine:
            self._affine = [
                RandomAffine(
                    degrees=180,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=5,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
            ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.images[i]
        peak = float(np.max(x))
        if peak > 0.0:
            x = x / peak
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        if self.augment:
            if self._affine is not None:
                for t in self._affine:
                    x = t(x)
            if self.gauss_sigma_max > 0.0:
                sigma = float(torch.empty(1).uniform_(0.0, self.gauss_sigma_max))
                x = x + sigma * torch.randn_like(x)
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return x, label

def stratified_split(N, rng_idx, train, test, val):
    rng = np.random.default_rng(rng_idx)
    suma = train + test + val
    if not (abs(suma - 1) < 1e-9 or abs(suma - 100) < 1e-9):
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
