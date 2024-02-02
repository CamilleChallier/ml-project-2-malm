import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGG19(nn.Module):
    def __init__(self, in_channels=1, num_classes=4) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.vgg = vgg19().features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=num_classes, bias=True),
        )

    def forward(self, X):
        out = self.in_conv(X)
        out = self.vgg(out)
        out = self.classifier(out)
        return out
