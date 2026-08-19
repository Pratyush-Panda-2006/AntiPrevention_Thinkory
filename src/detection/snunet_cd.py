import torch
import torch.nn as nn

from detection.snunet_modules import (
    SNUNetChannelAttention,
    SNUNetNestedConvBlock,
    SNUNetUpBlock,
)


class SNUNetCD(nn.Module):
    """
    SNUNet-CD architecture port for the current project.

    IMPORTANT:
        This implementation follows the reference architecture,
        but exposes ONLY the final fused prediction by default.

    Production interface:

        model(image_a, image_b)
            -> logits [B, 1, H, W]

    Debug interface:

        model(
            image_a,
            image_b,
            return_intermediates=True,
        )

            -> (final_logits, intermediate_dict)

    The debug mode exists only for architecture verification and
    does not change the production TrainingEngine interface.

    Reference characteristics preserved:

        - shared Siamese encoder
        - 5 feature levels
        - base channels = 32
        - dense nested decoder
        - concatenation-based temporal fusion
        - ECAM
        - final fused output
        - reference deepest-path asymmetry:
          x4_0A is not computed; x4_0B feeds x3_1

    Deep supervision:
        Not used as an auxiliary training loss in Model 3 v1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "SNUNetCD expects 3-channel RGB input."
            )

        # =====================================================
        # Reference channel configuration
        # =====================================================

        n1 = 32

        filters = [
            n1,
            n1 * 2,
            n1 * 4,
            n1 * 8,
            n1 * 16,
        ]

        self.filters = tuple(filters)

        # =====================================================
        # Shared pooling
        # =====================================================

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # =====================================================
        # Encoder
        #
        # One set of modules is shared between T1 and T2.
        # =====================================================

        self.conv0_0 = SNUNetNestedConvBlock(
            in_channels,
            filters[0],
            filters[0],
        )

        self.conv1_0 = SNUNetNestedConvBlock(
            filters[0],
            filters[1],
            filters[1],
        )

        self.Up1_0 = SNUNetUpBlock(
            filters[1]
        )

        self.conv2_0 = SNUNetNestedConvBlock(
            filters[1],
            filters[2],
            filters[2],
        )

        self.Up2_0 = SNUNetUpBlock(
            filters[2]
        )

        self.conv3_0 = SNUNetNestedConvBlock(
            filters[2],
            filters[3],
            filters[3],
        )

        self.Up3_0 = SNUNetUpBlock(
            filters[3]
        )

        self.conv4_0 = SNUNetNestedConvBlock(
            filters[3],
            filters[4],
            filters[4],
        )

        self.Up4_0 = SNUNetUpBlock(
            filters[4]
        )

        # =====================================================
        # Nested decoder — depth 1
        # =====================================================

        self.conv0_1 = SNUNetNestedConvBlock(
            filters[0] * 2 + filters[1],
            filters[0],
            filters[0],
        )

        self.conv1_1 = SNUNetNestedConvBlock(
            filters[1] * 2 + filters[2],
            filters[1],
            filters[1],
        )

        self.Up1_1 = SNUNetUpBlock(
            filters[1]
        )

        self.conv2_1 = SNUNetNestedConvBlock(
            filters[2] * 2 + filters[3],
            filters[2],
            filters[2],
        )

        self.Up2_1 = SNUNetUpBlock(
            filters[2]
        )

        self.conv3_1 = SNUNetNestedConvBlock(
            filters[3] * 2 + filters[4],
            filters[3],
            filters[3],
        )

        self.Up3_1 = SNUNetUpBlock(
            filters[3]
        )

        # =====================================================
        # Nested decoder — depth 2
        # =====================================================

        self.conv0_2 = SNUNetNestedConvBlock(
            filters[0] * 3 + filters[1],
            filters[0],
            filters[0],
        )

        self.conv1_2 = SNUNetNestedConvBlock(
            filters[1] * 3 + filters[2],
            filters[1],
            filters[1],
        )

        self.Up1_2 = SNUNetUpBlock(
            filters[1]
        )

        self.conv2_2 = SNUNetNestedConvBlock(
            filters[2] * 3 + filters[3],
            filters[2],
            filters[2],
        )

        self.Up2_2 = SNUNetUpBlock(
            filters[2]
        )

        # =====================================================
        # Nested decoder — depth 3
        # =====================================================

        self.conv0_3 = SNUNetNestedConvBlock(
            filters[0] * 4 + filters[1],
            filters[0],
            filters[0],
        )

        self.conv1_3 = SNUNetNestedConvBlock(
            filters[1] * 4 + filters[2],
            filters[1],
            filters[1],
        )

        self.Up1_3 = SNUNetUpBlock(
            filters[1]
        )

        # =====================================================
        # Nested decoder — depth 4
        # =====================================================

        self.conv0_4 = SNUNetNestedConvBlock(
            filters[0] * 5 + filters[1],
            filters[0],
            filters[0],
        )

        # =====================================================
        # ECAM
        #
        # Reference:
        #     ca  = ChannelAttention(128, ratio=16)
        #     ca1 = ChannelAttention(32, ratio=4)
        # =====================================================

        self.ca = SNUNetChannelAttention(
            filters[0] * 4,
            ratio=16,
        )

        self.ca1 = SNUNetChannelAttention(
            filters[0],
            ratio=4,
        )

        # x0_1 + x0_2 + x0_3 + x0_4
        # = 32 * 4 = 128 channels
        self.conv_final = nn.Conv2d(
            filters[0] * 4,
            num_classes,
            kernel_size=1,
        )

        self._initialize_weights()

    # =========================================================
    # Initialization
    # =========================================================

    def _initialize_weights(self):
        """
        Match the reference initialization for newly created
        SNUNet-CD layers.
        """

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.ConvTranspose2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        image_a,
        image_b,
        return_intermediates: bool = False,
    ):
        """
        Forward pass.

        Default:
            returns final logits tensor.

        Debug:
            returns:
                final_logits,
                intermediate_dict
        """

        # =====================================================
        # T1 encoder
        # =====================================================

        x0_0A = self.conv0_0(
            image_a
        )

        x1_0A = self.conv1_0(
            self.pool(x0_0A)
        )

        x2_0A = self.conv2_0(
            self.pool(x1_0A)
        )

        x3_0A = self.conv3_0(
            self.pool(x2_0A)
        )

        # IMPORTANT:
        # The reference does NOT compute x4_0A.
        #
        # We intentionally preserve that behavior.

        # =====================================================
        # T2 encoder
        # =====================================================

        x0_0B = self.conv0_0(
            image_b
        )

        x1_0B = self.conv1_0(
            self.pool(x0_0B)
        )

        x2_0B = self.conv2_0(
            self.pool(x1_0B)
        )

        x3_0B = self.conv3_0(
            self.pool(x2_0B)
        )

        x4_0B = self.conv4_0(
            self.pool(x3_0B)
        )

        # =====================================================
        # Nested decoder
        # =====================================================

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0A,
                    x0_0B,
                    self.Up1_0(x1_0B),
                ],
                dim=1,
            )
        )

        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0A,
                    x1_0B,
                    self.Up2_0(x2_0B),
                ],
                dim=1,
            )
        )

        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0A,
                    x0_0B,
                    x0_1,
                    self.Up1_1(x1_1),
                ],
                dim=1,
            )
        )

        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0A,
                    x2_0B,
                    self.Up3_0(x3_0B),
                ],
                dim=1,
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0A,
                    x1_0B,
                    x1_1,
                    self.Up2_1(x2_1),
                ],
                dim=1,
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0A,
                    x0_0B,
                    x0_1,
                    x0_2,
                    self.Up1_2(x1_2),
                ],
                dim=1,
            )
        )

        # Reference deepest path:
        # x3_1 receives x3_0A, x3_0B and Up4_0(x4_0B).
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0A,
                    x3_0B,
                    self.Up4_0(x4_0B),
                ],
                dim=1,
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0A,
                    x2_0B,
                    x2_1,
                    self.Up3_1(x3_1),
                ],
                dim=1,
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0A,
                    x1_0B,
                    x1_1,
                    x1_2,
                    self.Up2_2(x2_2),
                ],
                dim=1,
            )
        )

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0A,
                    x0_0B,
                    x0_1,
                    x0_2,
                    x0_3,
                    self.Up1_3(x1_3),
                ],
                dim=1,
            )
        )

        # =====================================================
        # ECAM
        # =====================================================

        out = torch.cat(
            [
                x0_1,
                x0_2,
                x0_3,
                x0_4,
            ],
            dim=1,
        )

        intra = torch.sum(
            torch.stack(
                [
                    x0_1,
                    x0_2,
                    x0_3,
                    x0_4,
                ],
                dim=0,
            ),
            dim=0,
        )

        ca1 = self.ca1(
            intra
        )

        out = self.ca(out) * (
            out
            + ca1.repeat(
                1,
                4,
                1,
                1,
            )
        )

        logits = self.conv_final(
            out
        )

        # =====================================================
        # Production interface
        # =====================================================

        if not return_intermediates:
            return logits

        # =====================================================
        # Debug interface
        # =====================================================

        intermediates = {
            "x0_0A": x0_0A,
            "x1_0A": x1_0A,
            "x2_0A": x2_0A,
            "x3_0A": x3_0A,
            "x0_0B": x0_0B,
            "x1_0B": x1_0B,
            "x2_0B": x2_0B,
            "x3_0B": x3_0B,
            "x4_0B": x4_0B,
            "x0_1": x0_1,
            "x1_1": x1_1,
            "x2_1": x2_1,
            "x3_1": x3_1,
            "x0_2": x0_2,
            "x1_2": x1_2,
            "x2_2": x2_2,
            "x0_3": x0_3,
            "x1_3": x1_3,
            "x0_4": x0_4,
            "nested_concat": out,
            "intra_attention_input": intra,
            "attention_coarse": ca1,
            "logits": logits,
        }

        return logits, intermediates