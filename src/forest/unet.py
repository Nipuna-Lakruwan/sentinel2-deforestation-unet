"""A small U-Net for per-pixel forest / not-forest."""
import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Encoder: 4 levels, each halves the image and doubles the channels (sees more context).
    Decoder: upsample back, concatenating the encoder's same-size features ("skip connections")
    so fine edges, such as the border of a clearing, survive the trip through the bottleneck.

    Input  (B, in_ch, H, W) with H, W divisible by 8.  Output (B, 1, H, W) logits for "forest".
    """

    def __init__(self, in_ch: int = 8, base: int = 24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([conv_block(in_ch, c[0]), conv_block(c[0], c[1]), conv_block(c[1], c[2])])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([conv_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out(x)
