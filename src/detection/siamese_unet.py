import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two convolution layers with BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Convolutional block followed by max pooling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)

        return features, pooled


class DecoderBlock(nn.Module):
    """
    Upsampling + skip connection + convolution block.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Safety for possible spatial-size differences.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class SharedEncoder(nn.Module):
    """
    Shared-weight encoder.

    T1 and T2 pass through the SAME encoder.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.block1 = EncoderBlock(
            in_channels,
            64,
        )

        self.block2 = EncoderBlock(
            64,
            128,
        )

        self.block3 = EncoderBlock(
            128,
            256,
        )

        self.block4 = EncoderBlock(
            256,
            512,
        )

        self.bottleneck = ConvBlock(
            512,
            1024,
        )

    def forward(self, x):
        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)
        skip4, x = self.block4(x)

        x = self.bottleneck(x)

        return (
            skip1,
            skip2,
            skip3,
            skip4,
            x,
        )


class SiameseUNet(nn.Module):
    """
    Siamese U-Net baseline for binary change detection.

    T1 and T2 share the same encoder weights.

    Encoder features from T1 and T2 are compared using
    absolute feature differences.

    The decoder reconstructs a pixel-level change mask.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
    ):
        super().__init__()

        if in_channels not in (2, 3):
            raise ValueError(
                "SiameseUNet expects 2-channel SAR or 3-channel RGB input."
            )

        # ---------------------------------------------------------
        # Shared Siamese encoder
        # ---------------------------------------------------------

        self.encoder = SharedEncoder(
            in_channels=in_channels
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        self.decoder4 = DecoderBlock(
            in_channels=1024,
            skip_channels=512,
            out_channels=512,
        )

        self.decoder3 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.decoder2 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder1 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        # ---------------------------------------------------------
        # Final pixel classifier
        # ---------------------------------------------------------

        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1,
        )

        nn.init.constant_(
            self.final.bias,
            -3.0,
        )

    def forward(self, image_a, image_b):

        # ---------------------------------------------------------
        # Shared encoder
        # ---------------------------------------------------------

        (
            a1,
            a2,
            a3,
            a4,
            a5,
        ) = self.encoder(image_a)

        (
            b1,
            b2,
            b3,
            b4,
            b5,
        ) = self.encoder(image_b)

        # ---------------------------------------------------------
        # Feature differences
        # ---------------------------------------------------------

        diff1 = torch.abs(a1 - b1)
        diff2 = torch.abs(a2 - b2)
        diff3 = torch.abs(a3 - b3)
        diff4 = torch.abs(a4 - b4)
        diff5 = torch.abs(a5 - b5)

        # ---------------------------------------------------------
        # Decode
        # ---------------------------------------------------------

        x = self.decoder4(
            diff5,
            diff4,
        )

        x = self.decoder3(
            x,
            diff3,
        )

        x = self.decoder2(
            x,
            diff2,
        )

        x = self.decoder1(
            x,
            diff1,
        )

        # ---------------------------------------------------------
        # Output logits
        # ---------------------------------------------------------

        return self.final(x)
