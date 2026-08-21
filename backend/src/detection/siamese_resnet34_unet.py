import torch
import torch.nn as nn
from torchvision.models import (
    resnet34,
    ResNet34_Weights,
)

from detection.siamese_unet import (
    ConvBlock,
    DecoderBlock,
)


class SharedResNet34Encoder(nn.Module):
    """
    Shared ImageNet-pretrained ResNet-34 encoder.

    T1 and T2 must pass through the SAME encoder instance.

    Feature resolutions for a 256x256 input:

        conv1  -> 1/2   -> 64 channels
        layer1 -> 1/4   -> 64 channels
        layer2 -> 1/8   -> 128 channels
        layer3 -> 1/16  -> 256 channels
        layer4 -> 1/32  -> 512 channels

    IMPORTANT:
        conv1 output is captured BEFORE maxpool.
    """

    def __init__(self, in_channels=3, sar_init_mode="average"):
        super().__init__()

        if in_channels not in (2, 3):
            raise ValueError(
                "ImageNet-pretrained ResNet-34 expects "
                "2-channel SAR or 3-channel RGB input."
            )

        backbone = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        # ---------------------------------------------------------
        # ResNet-34 stem
        # ---------------------------------------------------------

        self.conv1 = backbone.conv1

        if in_channels == 2:
            old_conv1 = self.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=False,
            )

            if sar_init_mode == "average":
                # Averages the ImageNet RGB weights across the channel dimension,
                # then duplicates that average for the 2 SAR channels.
                with torch.no_grad():
                    avg_weights = old_conv1.weight.mean(dim=1, keepdim=True)
                    self.conv1.weight.data.copy_(avg_weights.repeat(1, 2, 1, 1))
            elif sar_init_mode == "random":
                nn.init.kaiming_normal_(
                    self.conv1.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            else:
                raise ValueError(f"Unknown sar_init_mode: {sar_init_mode}")

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # ---------------------------------------------------------
        # ResNet-34 residual stages
        # ---------------------------------------------------------

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        # ---------------------------------------------------------
        # Stem
        # ---------------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Capture BEFORE maxpool.
        skip_conv1 = x

        # ---------------------------------------------------------
        # ResNet stages
        # ---------------------------------------------------------

        x = self.maxpool(x)

        skip_layer1 = self.layer1(x)
        skip_layer2 = self.layer2(skip_layer1)
        skip_layer3 = self.layer3(skip_layer2)
        bottleneck = self.layer4(skip_layer3)

        return (
            skip_conv1,
            skip_layer1,
            skip_layer2,
            skip_layer3,
            bottleneck,
        )


class SiameseResNet34UNet(nn.Module):
    """
    Model 2: Siamese U-Net with an ImageNet-pretrained ResNet-34
    shared encoder.

    Architecture:

        T1 ──┐
             ├── Shared ResNet-34
        T2 ──┘

        Absolute feature differences are computed at every
        encoder level.

        Resolutions:

            conv1       1/2   64
            layer1      1/4   64
            layer2      1/8   128
            layer3      1/16  256
            layer4      1/32  512

        Decoder:

            1/32 -> 1/16  + layer3
            1/16 -> 1/8   + layer2
            1/8  -> 1/4   + layer1
            1/4  -> 1/2   + conv1
            1/2  -> 1/1   no skip

        Output:

            [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        sar_init_mode="average",
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Shared pretrained encoder
        # ---------------------------------------------------------

        self.encoder = SharedResNet34Encoder(
            in_channels=in_channels,
            sar_init_mode=sar_init_mode,
        )

        # ---------------------------------------------------------
        # Decoder
        #
        # layer4: 512 @ 1/32
        # layer3: 256 @ 1/16
        # ---------------------------------------------------------

        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        # ---------------------------------------------------------
        # layer3 -> layer2
        #
        # 256 @ 1/16
        # layer2: 128 @ 1/8
        # ---------------------------------------------------------

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        # ---------------------------------------------------------
        # layer2 -> layer1
        #
        # 128 @ 1/8
        # layer1: 64 @ 1/4
        # ---------------------------------------------------------

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        # ---------------------------------------------------------
        # layer1 -> conv1
        #
        # 64 @ 1/4
        # conv1: 64 @ 1/2
        # ---------------------------------------------------------

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64,
        )

        # ---------------------------------------------------------
        # Final 1/2 -> 1/1 upsample
        #
        # There is intentionally NO full-resolution skip.
        # This is a documented architectural consequence of the
        # standard ResNet-34 stem.
        # ---------------------------------------------------------

        self.final_up = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2,
        )

        # ---------------------------------------------------------
        # Final pixel classifier
        # ---------------------------------------------------------

        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1,
        )

        # Same prior-bias initialization used by Model 1.
        nn.init.constant_(
            self.final.bias,
            -3.0,
        )

    def forward(self, image_a, image_b):

        # ---------------------------------------------------------
        # Shared Siamese encoder
        # ---------------------------------------------------------

        (
            a_conv1,
            a_layer1,
            a_layer2,
            a_layer3,
            a_layer4,
        ) = self.encoder(image_a)

        (
            b_conv1,
            b_layer1,
            b_layer2,
            b_layer3,
            b_layer4,
        ) = self.encoder(image_b)

        # ---------------------------------------------------------
        # Absolute feature differences
        # ---------------------------------------------------------

        diff_conv1 = torch.abs(
            a_conv1 - b_conv1
        )

        diff_layer1 = torch.abs(
            a_layer1 - b_layer1
        )

        diff_layer2 = torch.abs(
            a_layer2 - b_layer2
        )

        diff_layer3 = torch.abs(
            a_layer3 - b_layer3
        )

        diff_layer4 = torch.abs(
            a_layer4 - b_layer4
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        x = self.decoder4(
            diff_layer4,
            diff_layer3,
        )

        x = self.decoder3(
            x,
            diff_layer2,
        )

        x = self.decoder2(
            x,
            diff_layer1,
        )

        x = self.decoder1(
            x,
            diff_conv1,
        )

        # ---------------------------------------------------------
        # Final 1/2 -> 1/1 upsample
        # ---------------------------------------------------------

        x = self.final_up(x)

        # ---------------------------------------------------------
        # Output logits
        # ---------------------------------------------------------

        return self.final(x)