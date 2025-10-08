import torch.nn as nn

from utils.utils import initialize_weights_orthogonal


class SimpleCNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        cnn = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten())
        extra_layer = nn.Sequential(
            nn.Linear(64 * 1 * 1, out_dim),
            nn.BatchNorm1d(out_dim), nn.ReLU())

        initialize_weights_orthogonal(cnn)
        initialize_weights_orthogonal(extra_layer, std=0.1)
        self.backbone = nn.Sequential(cnn, extra_layer)

    def forward(self, x):
        feat = self.backbone(x)
        return feat
