from segment_anything_training.modeling.common import LayerNorm2d
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,stride=2, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,stride=2, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upChannel=nn.Conv2d(128,768,kernel_size=1,stride=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 =self.up(x2)
        x4 = self.upChannel(x3)
        return x4

# model = UNet()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# x = torch.randn(1, 3, 1024, 1024).to(device)
# print(model(x).shape)

