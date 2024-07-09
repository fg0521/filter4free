from torch import nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class PConv2d(nn.Module):
    def __init__(self, dim, n_div=4,kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class FilterSimulation(nn.Module):
    def __init__(self, training=False, channel=3):
        super(FilterSimulation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            PConv2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32,3,kernel_size=3,padding=1)
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
        x2 = F.interpolate(x2, mode='bilinear', scale_factor=2, align_corners=False)
        x2 = self.final_conv(x2)
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
        x1 = self.inc(x)    #[bs,32,256,256]
        x2 = self.down1(x1) #[bs,64,128,128]
        x3 = self.down2(x2) #[bs,128,64,64]
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def image2block(image, patch_size=448, padding=16):
    patches = []
    # 转换为tensor
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # 归一化
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)  # c h w

    _, H, W = image.shape
    # 上侧、左侧填充padding  右侧和下侧需要计算
    right_padding = padding if W % patch_size == 0 else padding + patch_size - (W % patch_size)
    bottom_padding = padding if H % patch_size == 0 else padding + patch_size - (H % patch_size)
    image = F.pad(image, (padding, right_padding, padding, bottom_padding), mode='replicate')
    row = (image.shape[1] - 2 * padding) // patch_size
    col = (image.shape[2] - 2 * padding) // patch_size
    # 从左到右 从上到下
    for y1 in range(padding, row * patch_size, patch_size):
        for x1 in range(padding, col * patch_size, patch_size):
            patch = image[:, y1 - padding:y1 + patch_size + padding, x1 - padding:x1 + patch_size + padding]
            patch = patch.unsqueeze(0)
            patches.append(patch)
    return patches, row, col


def infer(image, model, device, patch_size=448, batch=8, padding=16):
    img_input = cv2.imread(image) if isinstance(image, str) else image
    channel = 3  # 模型输出的通道数
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) if channel == 3 else cv2.cvtColor(img_input,
                                                                                             cv2.COLOR_BGR2GRAY)
    split_images, row, col = image2block(img_input, patch_size=patch_size, padding=padding)
    img_output = torch.zeros((row * patch_size, col * patch_size, channel), dtype=torch.float)
    with torch.no_grad():
        for i in range(0, len(split_images), batch):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            batch_output = model(batch_input.to(device))
            batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3, 1).cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                img_output[y:y + patch_size, x:x + patch_size] = output
    img_output = img_output[:img_input.shape[0], :img_input.shape[1]].numpy()
    mean = np.array([-2.12, -2.04, -1.80])
    std = np.array([4.36, 4.46, 4.44])
    img_output = (img_output - mean) / std
    img_output = np.clip(img_output * 255, a_min=0, a_max=255).astype(np.uint8)
    # img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
    return img_input, img_output
