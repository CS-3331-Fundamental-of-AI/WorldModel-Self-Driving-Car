import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Conv → BN → GELU block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)

class BEVJEPAEncoder2D(nn.Module):
    """
    2D JEPA Context/Target Encoder (replaces TokenMLPEncoder)
    - 4 CNN stages (Zhu JEPA topological equivalent)
    - Output: BEV tokens (B, HW, C)
    """
    def __init__(self, in_ch=3, base_dim=64):
        super().__init__()

        C = base_dim

        # -------- Stage 1 --------
        self.s1 = nn.Sequential(
            ConvBlock(in_ch, C),
            ConvBlock(C, C),
            ConvBlock(C, C),
        )

        # -------- Stage 2 --------
        self.s2 = nn.Sequential(
            nn.Conv2d(C, 2*C, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvBlock(2*C, 2*C),
            ConvBlock(2*C, 2*C),
        )

        # -------- Stage 3 --------
        self.s3 = nn.Sequential(
            nn.Conv2d(2*C, 4*C, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvBlock(4*C, 4*C),
            ConvBlock(4*C, 4*C),
        )

        # -------- Stage 4 --------
        self.s4 = nn.Sequential(
            nn.Conv2d(4*C, 8*C, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ConvBlock(8*C, 8*C),
            ConvBlock(8*C, 8*C),
        )

        self.out_dim = 8 * C

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns:
            tokens: (B, HW, C_out)
            (H', W')
        """
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        B, C, H, W = x.shape

        tokens = x.flatten(2).transpose(1, 2)   # (B, H * W, C_out)

        return tokens, (H, W)