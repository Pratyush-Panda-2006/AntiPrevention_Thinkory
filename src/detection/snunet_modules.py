import torch
import torch.nn as nn


class SNUNetNestedConvBlock(nn.Module):
    """
    Convolutional block used by the reference SNUNet-CD implementation.

    Structure:

        Conv2d
        BatchNorm
        ReLU
        Conv2d
        BatchNorm
        residual addition
        ReLU

    The residual connection is taken after the first convolution,
    matching the reference implementation.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.activation = nn.ReLU(
            inplace=True
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.bn1 = nn.BatchNorm2d(
            mid_channels
        )

        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

    def forward(self, x):
        x = self.conv1(x)

        # Reference implementation takes the residual
        # after conv1 and before BN/ReLU.
        identity = x

        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        output = self.activation(
            x + identity
        )

        return output


class SNUNetUpBlock(nn.Module):
    """
    Reference SNUNet-CD upsampling block.

    The reference uses ConvTranspose2d rather than bilinear
    interpolation for the default configuration.
    """

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        return self.up(x)


class SNUNetChannelAttention(nn.Module):
    """
    Channel attention used by the reference SNUNet-CD ECAM.

    Uses both:
        - adaptive average pooling
        - adaptive max pooling

    followed by a shared two-layer 1x1 convolutional MLP
    and sigmoid gating.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: int = 16,
    ):
        super().__init__()

        if in_channels < ratio:
            raise ValueError(
                f"in_channels ({in_channels}) must be >= "
                f"ratio ({ratio})."
            )

        reduced_channels = in_channels // ratio

        if reduced_channels < 1:
            raise ValueError(
                "Channel attention reduction produced "
                "zero channels."
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Conv2d(
            reduced_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(
            self.relu(
                self.fc1(
                    self.avg_pool(x)
                )
            )
        )

        max_out = self.fc2(
            self.relu(
                self.fc1(
                    self.max_pool(x)
                )
            )
        )

        attention = avg_out + max_out

        return self.sigmoid(
            attention
        )