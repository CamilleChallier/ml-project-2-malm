import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Lazy modules are a new feature under heavy development so changes to the API or functionality can happen at any moment.",
)

import torch
import torch.nn as nn

"""
Code adapted from: https://github.com/neergaard/utime-pytorch/blob/main/models/utime.py
"""


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=5,
        kernel_size=3,
        dilation=1,
        activation="relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1
        ) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        filters=[16, 32, 64, 128],
        in_channels=5,
        maxpool_kernels=[10, 8, 6, 4],
        kernel_size=5,
        dilation=2,
    ):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])

        self.maxpools = nn.ModuleList(
            [nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)]
        )

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)
        # Bottom part
        encoded = self.bottom(x)
        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(
        self,
        filters=[128, 64, 32, 16],
        upsample_kernels=[4, 6, 8, 10],
        in_channels=256,
        out_channels=5,
        kernel_size=5,
    ):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_kernels[k]),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                activation='relu',
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(
            self.upsamples, self.blocks, shortcuts[::-1]
        ):
            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)
        return z


class Utime(nn.Module):
    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 4,
        decoder_out_channels: int = 5,
        filters=[16, 32, 64, 128],
        maxpool_kernels=[10, 8, 6, 4],
        kernel_size=5,
        dilation=2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.decoder_out_channels = decoder_out_channels
        self.filters = filters
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.encoder = Encoder(
            filters=filters,
            in_channels=in_channels,
            maxpool_kernels=maxpool_kernels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.decoder = Decoder(
            filters=filters[::-1],
            upsample_kernels=maxpool_kernels[::-1],
            in_channels=filters[-1] * 2,
            kernel_size=kernel_size,
            out_channels=decoder_out_channels,
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(
                out_features=out_channels,
                bias=True,
            ),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        # Run through encoder
        z, shortcuts = self.encoder(x)
        # Run through decoder
        z = self.decoder(z, shortcuts)
        # Run dense modeling
        z = self.dense(z)
        return z
