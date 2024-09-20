import copy
import os

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models, transforms

from utils import to_pil


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


class FilterSimulation1(nn.Module):
    """
    版本1：二维卷积+二维反卷积
    注：小分辨率图像容易出现棋盘效应问题
    """

    def __init__(self, channel=3, training=False):
        super(FilterSimulation1, self).__init__()
        self.name = 'FilterSimulation1'
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FilterSimulation2(nn.Module):
    """
    版本2：二维卷积+双线性插值
    注：解决棋盘效应，考虑压缩模型大小
    """

    def __init__(self, channel=3, training=False):
        self.name = 'FilterSimulation2'
        super(FilterSimulation2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
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
        )
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        x2 = F.interpolate(x2, mode='bilinear', size=x.shape[2:], align_corners=False)
        x2 = self.final_conv(x2)
        return x2


class FilterSimulation3(nn.Module):
    """
    版本2：可分离通道卷积+双线性插值
    注：压缩模型大小 380K->80K

    """

    def __init__(self, channel=3, training=False):
        super(FilterSimulation3, self).__init__()
        self.name = 'FilterSimulation3'
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
            nn.Conv2d(channel, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, groups=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        x2 = F.interpolate(x2, mode='bilinear', size=x.shape[2:], align_corners=False)
        x2 = self.final_conv(x2)
        return x2


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


class ResidualBlock(nn.Module):
    """残差块，包含两个卷积层和跳跃连接"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


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


class UNet4UCM(nn.Module):
    def __init__(self, channel=4):
        super(UNet4UCM, self).__init__()
        self.inc = (DoubleConv(channel, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.up2 = (Up(128))
        self.up3 = (Up(64))
        self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, content_features, color_features):
        x = torch.cat((content_features, color_features), dim=1)  # [bs, 4, 256, 256]
        x1 = self.inc(x)  # [bs,32,256,256]
        x2 = self.down1(x1)  # [bs,64,128,128]
        x3 = self.down2(x2)  # [bs,128,64,64]
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class UCM(nn.Module):
    def __init__(self, k=16, continue_train=False) -> None:
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.k = k
        self.sNet = UNet4UCM()
        self.cNet = UNet4UCM()
        self.D = nn.Linear(in_features=1000, out_features=k * k)  # image content info
        self.S = nn.Linear(in_features=1000, out_features=k * k)  # image color info
        # 反卷积层，用于上采样
        self.D_conv_transpose = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道1，输出通道64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道64，输出通道32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 输入通道32，输出通道16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 输入通道16，输出通道1
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        )

        self.S_conv_transpose = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道1，输出通道64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道64，输出通道32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 输入通道32，输出通道16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 输入通道16，输出通道1
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        )

        self._initialize_weights()
        self.continue_train = continue_train
        if continue_train:
            # 冻结backbone层
            self.backbone.requires_grad = False
            self.backbone.eval()

    def _initialize_weights(self):
        [nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') for layer in self.D_conv_transpose
         if layer._get_name() == 'Conv2d']
        [nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') for layer in self.S_conv_transpose
         if layer._get_name() == 'Conv2d']

    def forward(self, org_img, filter_img=None, filter=None):
        if filter_img is not None:
            # train
            bs = org_img.shape[0]
            out1 = self.backbone(org_img)
            out2 = self.backbone(filter_img)
            d1 = self.D(out1).view(-1, 1, self.k, self.k)  # [bs, 256]
            d1 = self.D_conv_transpose(d1)
            d2 = self.S(out2).view(-1, 1, self.k, self.k)  # [bs, 256]
            d2 = self.S_conv_transpose(d2)
            s1 = self.D(out1).view(-1, 1, self.k, self.k)  # [bs, 256]
            s1 = self.D_conv_transpose(s1)
            s2 = self.S(out2).view(-1, 1, self.k, self.k)  # [bs, 256]
            s2 = self.S_conv_transpose(s2)
            if self.continue_train:
                # 仅训练一组特定的色彩
                s1 = s1.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)
                s2 = s2.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)
            #     s1 = s2.mean(dim=0, keepdim=True).repeat(bs, 1,1,1)
            org_content = self.sNet(org_img, d1)  # 去色的原始图像
            f_content = self.sNet(filter_img, d2)  # 去色的滤镜图像
            Y = self.cNet(f_content, s1)  # 去色的滤镜图像+原始图像色彩
            Y_i = self.cNet(org_content, s2)
        else:
            # infer
            bs = org_img.shape[0]
            filter = filter.repeat(bs, 1, 1, 1)
            out1 = self.backbone(org_img)
            d1 = self.D(out1).view(bs, 1, self.k, self.k)
            d1 = self.D_conv_transpose(d1)
            org_content = self.sNet(org_img, d1)  # 去色的原始图像
            Y = self.cNet(org_content, filter)  # 去色的滤镜图像+原始图像色彩
            f_content, s1, s2, Y_i = None, None, None, None

        return org_content, f_content, s1, s2, Y, Y_i


class Encoder(nn.Module):
    def __init__(self, k=16, training=False):
        super(Encoder, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.k = k
        self.D = nn.Linear(in_features=1000, out_features=k * k)  # image content info
        self.S = nn.Linear(in_features=1000, out_features=k * k)  # image color info
        # 反卷积层，用于上采样
        self.D_conv_transpose = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道1，输出通道64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道64，输出通道32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 输入通道32，输出通道16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 输入通道16，输出通道1
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        )

        self.S_conv_transpose = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道1，输出通道64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道64，输出通道32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 输入通道32，输出通道16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 输入通道16，输出通道1
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        )
        if training:
            self._initialize_weights()
            self.backbone.requires_grad = False
            self.backbone.eval()

    def _initialize_weights(self):
        [nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') for layer in self.D_conv_transpose
         if layer._get_name() == 'Conv2d']
        [nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') for layer in self.S_conv_transpose
         if layer._get_name() == 'Conv2d']

    def forward(self, image):
        x = self.backbone(image)
        d = self.D(x).view(-1, 1, self.k, self.k)  # content feature
        d = self.D_conv_transpose(d)
        r = self.S(x).view(-1, 1, self.k, self.k)  # style feature
        r = self.S_conv_transpose(r)
        return d, r


class Shader(nn.Module):
    def __init__(self, channel=4):
        super(Shader, self).__init__()
        self.inc = (DoubleConv(channel, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.up2 = (Up(128))
        self.up3 = (Up(64))
        self.outc = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, image, features):
        x = torch.cat((image, features), dim=1)  # [bs, 4, 256, 256]
        x1 = self.inc(x)  # [bs,32,256,256]
        x2 = self.down1(x1)  # [bs,64,128,128]
        x3 = self.down2(x2)  # [bs,128,64,64]
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def convert_checkpoint():
    pth1, pth2, pth3 = {}, {}, {}
    for k, v in pth.items():
        if k.startswith('sNet'):
            pth1[k.replace('sNet.', '')] = v
        elif k.startswith('cNet'):
            pth2[k.replace('cNet.', '')] = v
        else:
            pth3[k] = v
    torch.save({'encoder':pth3,'sNet':pth1,'cNet':pth2}, 'pretrained_model/pretrained_model_convert.pth')

if __name__ == '__main__':
    # device = torch.device('mps')
    # input = torch.rand((1, 3, 256, 256))
    # input = input.to(device)
    # model = UCM()
    # model.to(device)
    # org_content, f_content, s1, s2, Y, Y_i = model(input, input)
    # print(Y.shape, Y_i.shape)
    # torch.save(model.state_dict(), './test.pth')

    # torch.save(model.state_dict(),'/Users/maoyufeng/Downloads/11.pth')
    # print(output.shape)

    # import cv2
    # img1= cv2.imread('/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/train/17064945487403_org.jpg')
    # img2= cv2.imread('/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/train/17064945487403.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    # print(img1)

    pth = torch.load('pretrained_model/UCM_loss_0.2515.pth', map_location='cpu')


    pth1, pth2, pth3 = {}, {}, {}
    for k, v in pth.items():
        if k.startswith('sNet'):
            pth1[k.replace('sNet.', '')] = v
        elif k.startswith('cNet'):
            pth2[k.replace('cNet.', '')] = v
        else:
            pth3[k] = v
    encoder = Encoder()
    encoder.load_state_dict(pth3)
    encoder.eval()
    sNet = Shader()
    sNet.load_state_dict(pth1)
    sNet.eval()
    cNet = Shader()
    cNet.load_state_dict(pth2)
    cNet.eval()
    from PIL import Image
    image1 = Image.open('/Users/maoyufeng/Downloads/iShot_2024-09-11_09.12.26.jpg').convert('RGB')
    image2 = Image.open('/Users/maoyufeng/Downloads/iShot_2024-09-11_09.12.07.jpg').convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)

    d1,r1 = encoder(image1)
    d2,r2 = encoder(image2)

    content1 = sNet(image1,d1)
    content2 = sNet(image2,d2)

    y1 = cNet(content1,r2)
    y2 = cNet(content2,r1)

    y1 = to_pil(y1.squeeze(0))
    y2 = to_pil(y2.squeeze(0))

    y1.show()
    y2.show()



    new_pth = {
        'encoder': pth3,
        'sNet': pth1,
        'cNet': pth2
    }
    torch.save(new_pth, 'pretrained_model/pretrained_model_convert.pth')


