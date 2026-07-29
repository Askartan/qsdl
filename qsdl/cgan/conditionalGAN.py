import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, input_dim, output_dim):
        super().__init__()
        self.gene = nn.Sequential(
            nn.Linear(z_dim, input_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gene(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.001
asd
adasd
