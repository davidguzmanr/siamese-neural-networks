import torch
from torch import Tensor
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
    Siamese network used in "Siamese Neural Networks for One-shot Image Recognition".
    
    References
    - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # [m, 1, 105, 105] -> [m, 64, 96, 96]
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10),
            # [m, 64, 96, 96] -> [m, 64, 96, 96]
            nn.ReLU(),
            # [m, 64, 96, 96] -> [m, 64, 48, 48]
            nn.MaxPool2d(kernel_size=2),
            # [m, 64, 48, 48] -> [m, 128, 42, 42]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            # [m, 128, 42, 42] -> [m, 128, 42, 42]
            nn.ReLU(),
            # [m, 128, 42, 42] -> [m, 128, 21, 21]
            nn.MaxPool2d(kernel_size=2),
            # [m, 128, 21, 21] -> [m, 128, 18, 18]
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            # [m, 128, 18, 18] -> [m, 128, 18, 18]
            nn.ReLU(),
            # [m, 128, 18, 18] -> [m, 128, 9, 9]
            nn.MaxPool2d(kernel_size=2),
            # [m, 128, 9, 9] -> [m, 256, 6, 6]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            # [m, 256, 6, 6] -> [m, 256, 6, 6]
            nn.ReLU(),
        )

        # [m, 256, 6, 6] -> [m, 9216]
        self.flatten = nn.Flatten()

        # [m, 9216] -> [m, 4096]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.Sigmoid()
        )

        # [m, 4096] -> [m, 1]
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1),
            # nn.Sigmoid()
        )

    def forward_one_image(self, x: Tensor) -> Tensor:
        # [m, 1, 105, 105] -> [m, 256, 6, 6]
        x = self.conv(x)
        # [m, 256, 6, 6] -> [m, 9216]
        x = torch.flatten(x, start_dim=1)
        # [m, 9216] -> [m, 4096]
        x = self.fc1(x)

        return x

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.forward_one_image(x1)
        x2 = self.forward_one_image(x2)
        
        # L1 distance
        x = torch.abs(x1 - x2)
        x = self.fc2(x)

        return x