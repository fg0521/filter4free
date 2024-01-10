import os
import torch.nn.functional as F
import torch
import torch.nn as nn

# class FilmMask(nn.Module):
#     """
#     去色罩
#     AdamW: lr=1e-4
#     loss: MSELoss+ChiSquareLoss
#     训练数据: 10张图片
#     训练通道: LAB
#     epoch: 100
#     """
#
#     def __init__(self,training=False):
#         super(FilmMask, self).__init__()
#         self.encoder_r = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.encoder_g = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.encoder_b = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.decoder_r = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.decoder_g = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.decoder_b = nn.Sequential(
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         if training:
#             for net in ['self.encoder_r','self.encoder_g','self.encoder_b',
#                         'self.decoder_r', 'self.decoder_g', 'self.decoder_b']:
#                 for n in eval(net):
#                     if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
#                         nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, x):
#         # 编码器 解码器
#         r = x[:, 0, :, :].unsqueeze(1)
#         g = x[:, 1, :, :].unsqueeze(1)
#         b = x[:, 2, :, :].unsqueeze(1)
#         r = self.decoder_r(self.encoder_r(r))
#         g = self.decoder_g(self.encoder_g(g))
#         b = self.decoder_b(self.encoder_b(b))
#         x = torch.cat((r, g, b), dim=1)
#         return x


class FilterSimulation(nn.Module):
    """
    滤镜模拟
    AdamW: lr=0.002
    loss: MSELoss+EMDLoss
    训练数据: 64张图片
    训练通道: RGB
    epoch: 100
    """

    def __init__(self, training=False,channel=3):
        super(FilterSimulation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channel, kernel_size=2, stride=2),
        )
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)
        # 解码器
        x2 = self.decoder(x1)
        return x2



class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)


class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)


class FilterSimulationLarge(nn.Module):
    def __init__(self):
        super(FilterSimulationLarge, self).__init__()
        self.encoder_inc = DoubleConvBlock(3, 16)
        self.encoder_down1 = DownBlock(16, 32)
        self.encoder_down2 = DownBlock(32, 64)
        self.encoder_bridge_down = BridgeDown(64, 128)
        self.awb_decoder_bridge_up = BridgeUP(128, 64)
        self.awb_decoder_up1 = UpBlock(64, 32)
        self.awb_decoder_up2 = UpBlock(32, 16)
        self.awb_decoder_out = OutputBlock(16, 3)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x4 = self.encoder_down2(x2)
        x5 = self.encoder_bridge_down(x4)
        x_awb = self.awb_decoder_bridge_up(x5)
        x_awb = self.awb_decoder_up1(x_awb, x4)
        x_awb = self.awb_decoder_up2(x_awb, x2)
        awb = self.awb_decoder_out(x_awb, x1)
        return awb


class FilterNet(nn.Module):
    def __init__(self):
        super(FilterNet, self).__init__()
        self.encoder_inc = DoubleConvBlock(3, 32)
        self.encoder_down1 = DownBlock(32, 64)
        self.encoder_bridge_down = BridgeDown(64, 128)
        self.awb_decoder_bridge_up = BridgeUP(128, 64)
        self.awb_decoder_up1 = UpBlock(64, 32)
        self.awb_decoder_out = OutputBlock(32, 3)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_bridge_down(x2)
        x_out = self.awb_decoder_bridge_up(x3)
        x_out = self.awb_decoder_up1(x_out, x2)
        out = self.awb_decoder_out(x_out, x1)
        return out





if __name__ == '__main__':
    input = torch.rand((1,3,448,448))
    model = FilterSimulation()
    print(model(input).shape)

