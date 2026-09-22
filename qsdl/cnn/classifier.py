from torch import nn


class StateClassifierCNN(nn.Module):
    """Oryginalna, prosta architektura z repo (3 bloki konwolucyjne, ~kilkadziesiat tys. parametrow).
    Pasuje do checkpointow trenowanych domyslnym cnn.ipynb (np. clean.pt / noisy.pt)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,7)
        )

    def forward(self, x):
        return self.net(x)


class StateClassifierCNNv2(nn.Module):
    """Architektura Pjotra (checkpoint v2_train_noisy_19999.pt) - 4 bloki Conv-BN-ReLU x2 + MaxPool,
    head z AdaptiveAvgPool2d(4) i dwiema warstwami liniowymi (~2.22M parametrow).
    Zweryfikowane 1:1 wzgledem checkpointu: load_state_dict(strict=True) -> "All keys matched successfully"."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
