import torch.nn.functional as F
import torch
import torch.nn as nn



class PConv2d(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=kernel_size,
                                       stride=1, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class FilterSimulation4(nn.Module):
    def __init__(self, training=False, channel=3):
        super(FilterSimulation4, self).__init__()
        self.name = 'FilterSimulation4'
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = self.encoder(x)    # 编码器
        x2 = self.decoder(x1)   # 解码器
        x2 = F.interpolate(x2, mode='bilinear', scale_factor=2, align_corners=False)
        x2 = self.final_conv(x2)
        # x2 = (1.0 - temp) * x + temp * x2 # 引入温度系数 来控制图像变化
        return x2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            PConv2d(mid_channels),
            nn.ReLU(inplace=True)
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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(in_channels, in_channels // 2)
        self.conv2 = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.conv1(self.up(x1))
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class UNet(nn.Module):
    def __init__(self, channel=3):
        super(UNet, self).__init__()
        self.name = 'UNet'
        self.inc = (DoubleConv(channel, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.up2 = (Up(128))
        self.up3 = (Up(64))
        self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # [bs,32,256,256]
        x2 = self.down1(x1)  # [bs,64,128,128]
        x3 = self.down2(x2)  # [bs,128,64,64]
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits






