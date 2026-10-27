from torch import nn

class StateClassifierCNN(nn.Module):
    """4 bloki Conv-BN-ReLU x2 + MaxPool,
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
