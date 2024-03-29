import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from torchvision import models, transforms
from kornia.geometry.transform import resize
from kornia.enhance.normalize import Normalize
from torchvision.models import efficientnet_b0




class FilterSimulationFast(nn.Module):
    def __init__(self, channel=3,training=False):
        super(FilterSimulationFast, self).__init__()
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
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(64, 32 * (2 ** 2),kernel_size=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, channel, kernel_size=3, padding=1),
        )
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = F.pad(x, (16, 16, 16, 16), mode='replicate')
        x = self.encoder(x)
        x = self.decoder(x)
        x = x[:, :, 16:-16, 16:-16]  # 调整padding以移除额外边缘
        return x




class FilterSimulation(nn.Module):
    """
    滤镜模拟
    AdamW: lr=0.002
    loss: L1Loss+RGBLoss
    训练数据: 150张图片
    训练通道: RGB
    epoch: 150
    """

    def __init__(self, training=False, channel=3):
        super(FilterSimulation, self).__init__()
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
            # nn.ConvTranspose2d(32, channel, kernel_size=2, stride=2)
        )
        self.final_conv = nn.Conv2d(32,3,kernel_size=3,padding=1)
        if training:
            for net in ['self.encoder', 'self.decoder']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x,temp=1.0):
        # 编码器
        x1 = self.encoder(x)
        # 解码器
        x2 = self.decoder(x1)
        x2 = F.interpolate(x2, mode='bilinear', size=x.shape[2:], align_corners=False)
        x2 = self.final_conv(x2)
        # 引入温度系数 来控制图像变化
        x2 = (1.0 - temp) * x + temp * x2
        return x2





class FilterSimulationConvert(nn.Module):
    """
    滤镜模拟
    AdamW: lr=0.002
    loss: L1Loss+RGBLoss
    训练数据: 150张图片
    训练通道: RGB
    epoch: 150
    """

    def __init__(self):
        super(FilterSimulationConvert, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32,3,kernel_size=3,padding=1)


    def forward(self, x,temp):
        # 编码器
        x1 = self.encoder(x)
        # 解码器
        x1 = self.decoder(x1)
        x1 = F.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=False)
        x1 = self.final_conv(x1)
        # 引入温度系数 来控制图像变化
        x1 = (1.0 - temp) * x + temp * x1
        return x1


"""
https://arxiv.org/abs/2303.13511
Neural Preset for Color Style Transfer
"""

class DNCM(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k

    def forward(self, I, T):
        bs, _, H, W = I.shape   # [b,c,h,w]
        x = torch.flatten(I, start_dim=2).transpose(1, 2)   # [b,h*w,c]
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q  # [b,h*w,c]
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)    # [b,c,h,w]
        return out


class Encoder(nn.Module):
    def __init__(self, sz, k) -> None:
        super().__init__()
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.D = nn.Linear(in_features=1000, out_features=k * k)   # image content info
        self.S = nn.Linear(in_features=1000, out_features=k * k)    # image color info
        self.sz = sz

    def forward(self, I):
        I_theta = resize(I, self.sz, interpolation='bilinear')
        out = self.backbone(self.normalizer(I_theta))
        d = self.D(out)
        s = self.S(out)
        return d, s



# class DNCM(nn.Module):
#     def __init__(self, feature_dim=256, img_channels=3, img_size=256):
#         super(DNCM, self).__init__()
#         self.img_channels = img_channels
#         self.img_size = img_size
#
#         # 扩展特征至空间维度
#         self.fc = nn.Linear(feature_dim, img_size * img_size)  # 全连接层
#         # 特征图调整通道数
#         self.feature_conv = nn.Conv2d(1, img_channels, kernel_size=3, padding=1)
#         # 融合特征和图像
#         self.fusion_conv1 = nn.Conv2d(img_channels * 2, img_channels, kernel_size=3, padding=1)
#         self.fusion_conv2 = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
#         # 输出层调整，确保输出与输入图像大小一致
#         self.output_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
#
#         nn.init.kaiming_uniform_(self.feature_conv.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.fusion_conv1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.fusion_conv2.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.output_conv.weight, mode='fan_in', nonlinearity='relu')
#
#     def forward(self, img, features):
#         bs = img.shape[0]
#
#         # 扩展特征至空间维度并重塑为图像形状
#         spatial_features = self.fc(features).view(bs, 1, self.img_size, self.img_size)
#         spatial_features = F.relu(self.feature_conv(spatial_features))
#
#         # 将特征图和原图在通道维度上合并
#         merged = torch.cat([img, spatial_features], dim=1)
#
#         # 通过卷积层融合信息
#         merged = F.relu(self.fusion_conv1(merged))
#         merged = F.relu(self.fusion_conv2(merged))
#
#         # 输出层
#         output = self.output_conv(merged)
#         return output


if __name__ == '__main__':

    input = torch.rand(24, 3, 256, 256)
    input2 = torch.rand(24,256)
    # 1595998208
    # 1551040512

    model = DNCM()   # 514523136

    # model = FilterSimulation(training=True)         # 2025848832

    # flops = FlopCountAnalysis(model, (input,1))
    out = model.forward(input,input2)
    print(out.shape)
    # input_image = torch.rand(1, 3, 256, 256)
    # encoder = Encoder(sz=256,k=16)
    # content,color= encoder(input_image)
    # print(color.shape)
    # print(content.shape)

    # org_tensor = torch.rand((1,3,3,3))
    # res_tensor = torch.rand((1,3,3,3))
    #
    # print(org_tensor)
    # print(res_tensor)
    #
    # for temp in [0.0,0.5,1.0]:
    #     res = (1-temp)*org_tensor+temp*res_tensor
    #     print(res)
