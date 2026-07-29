from torch import nn


class StateClassifierCNN(nn.Module):
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
