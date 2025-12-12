import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
# MobileNet v1 Block (Depthwise 3×3 + Pointwise 1×1)
# ------------------------------------------------------
class MobileNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)

        self.pointwise = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x


# ------------------------------------------------------
# MobileNet-Style JEPA BEV Encoder (Tier-1)
# ------------------------------------------------------
class BEVJEPAEncoder2D(nn.Module):
    def __init__(self, base_channels=[16, 32, 64, 128], width_mult=0.5):
        super().__init__()

        # width multiplier (MobileNet α)
        C = [int(c * width_mult) for c in base_channels]

        self.s1 = nn.Sequential(
            MobileNetBlock(3,  C[0], stride=1),
            MobileNetBlock(C[0], C[0], stride=1),
        )

        self.s2 = nn.Sequential(
            MobileNetBlock(C[0], C[1], stride=2),
            MobileNetBlock(C[1], C[1], stride=1),
        )

        self.s3 = nn.Sequential(
            MobileNetBlock(C[1], C[2], stride=2),
            MobileNetBlock(C[2], C[2], stride=1),
        )

        self.s4 = nn.Sequential(
            MobileNetBlock(C[2], C[3], stride=2),
            MobileNetBlock(C[3], C[3], stride=1),
        )

        self.out_dim = C[3]

    def forward(self, x):
        x = self.s1(x)   # (B, C1, H,   W)
        x = self.s2(x)   # (B, C2, H/2, W/2)
        x = self.s3(x)   # (B, C3, H/4, W/4)
        x = self.s4(x)   # (B, C4, H/8, W/8)

        # JEPA EXPECTS: return (features, (H, W))
        H, W = x.shape[-2], x.shape[-1]

        return x, (H, W)