import torch
import torch.nn as nn
import torch.nn.functional as F



class FilmMask(nn.Module):
    """
    去色罩
    AdamW: lr=1e-4
    loss: MSELoss+ChiSquareLoss
    训练数据: 10张图片
    训练通道: LAB
    epoch: 100
    """
    def __init__(self):
        super(FilmMask, self).__init__()
        self.encoder_r = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_g = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_r = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_g = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_b = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # 编码器 解码器
        r = x[:,0,:,:].unsqueeze(1)
        g = x[:,1,:,:].unsqueeze(1)
        b = x[:,2,:,:].unsqueeze(1)
        r = self.decoder_r(self.encoder_r(r))
        g = self.decoder_g(self.encoder_g(g))
        b = self.decoder_b(self.encoder_b(b))
        x = torch.cat((r,g,b),dim=1)
        return x

class Olympus(nn.Module):
    """
    奥林巴斯浓郁色调
    AdamW: lr=0.002
    loss: MSELoss+ChiSquareLoss
    训练数据: 64张图片
    训练通道: RGB
    epoch: 100
    """
    def __init__(self):
        super(Olympus, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)
        # 中间层
        x2 = self.middle(x1)
        # 解码器
        x3 = self.decoder(x2)
        return x3


class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(3, 64)
        self.layer2_conv = double_conv2d_bn(64, 128)
        self.layer3_conv = double_conv2d_bn(128, 64)
        # self.layer4_conv = double_conv2d_bn(64, 64)
        self.layer5_conv = nn.Conv2d(64, 3, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        # self.deconv2 = deconv2d_bn(64, 64)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        # pool2 = F.max_pool2d(conv2, 2)

        convt1 = self.deconv1(conv2)
        concat1 = torch.cat([convt1, conv1], dim=1)
        conv3 = self.layer3_conv(concat1)

        # convt2 = self.deconv2(conv3)
        # concat2 = torch.cat([convt2, conv1], dim=1)
        # conv4 = self.layer4_conv(concat2)

        conv5 = self.layer5_conv(conv3)
        outp = self.sigmoid(conv5)
        return outp





if __name__ == '__main__':
    model = FilmMaskBase()
    inp = torch.rand(4, 3, 512, 512)
    outp = model(inp)
    print(outp.shape)