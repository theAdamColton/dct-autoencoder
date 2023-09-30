import torch.nn as nn

class ResidBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                )

    def forward(self, x):
        return self.block(x) + x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()
        self.layers = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                )

    def forward(self, x):
        return self.layers(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()
        self.layers = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, depth: int, image_channels: int = 3, channel_mult: int = 4):
        super().__init__()
        layers = []

        for i in range(1, depth+1):
            if i == 1:
                in_channels = image_channels
            else:
                in_channels = channel_mult**i

            out_channels = channel_mult **(i+1)

            layers.append(DownsampleBlock(in_channels, out_channels))
            layers.append(ResidBlock(out_channels, out_channels))
            in_channels = out_channels
            out_channels = in_channels * channel_mult

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, depth: int, image_channels: int=3, channel_mult:int=4):
        super().__init__()

        layers = []
        for i in range(depth, 0, -1):
            if i == 1:
                out_channels = image_channels
            else:
                out_channels = channel_mult ** i

            in_channels = channel_mult ** (i+1)

            layers.append(UpsampleBlock(in_channels, out_channels))
            layers.append(ResidBlock(out_channels, out_channels))
            in_channels = out_channels
            out_channels = in_channels // channel_mult

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

