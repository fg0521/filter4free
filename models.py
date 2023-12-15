import os
import torch.nn.functional as F
import torch
import torch.nn as nn

class FilmMask(nn.Module):
    """
    去色罩
    AdamW: lr=1e-4
    loss: MSELoss+ChiSquareLoss
    训练数据: 10张图片
    训练通道: LAB
    epoch: 100
    """

    def __init__(self,training=False):
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

        if training:
            for net in ['self.encoder_r','self.encoder_g','self.encoder_b',
                        'self.decoder_r', 'self.decoder_g', 'self.decoder_b']:
                for n in eval(net):
                    if n._get_name() in ['Conv2d', 'ConvTranspose2d']:
                        nn.init.kaiming_uniform_(n.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 编码器 解码器
        r = x[:, 0, :, :].unsqueeze(1)
        g = x[:, 1, :, :].unsqueeze(1)
        b = x[:, 2, :, :].unsqueeze(1)
        r = self.decoder_r(self.encoder_r(r))
        g = self.decoder_g(self.encoder_g(g))
        b = self.decoder_b(self.encoder_b(b))
        x = torch.cat((r, g, b), dim=1)
        return x


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
            # nn.MaxPool2d(kernel_size=1, stride=1)
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




if __name__ == '__main__':
    # input = torch.rand((3,224,224))
    # model = FilterSimulationLarge()
    # print(model(input).shape)
    model = FilterSimulation(channel=3)
    # pth = torch.load(f'static/checkpoints/fuji/eterna-bleach-bypass/best-lab.pth', map_location=torch.device('cpu'))
    # # print(pth)
    # new_pth = {}
    # for k, v in pth.items():
    #     if 'middle.0.weight' == k:
    #         new_pth['encoder.5.weight'] = v
    #     elif 'middle.0.bias' == k:
    #         new_pth['encoder.5.bias'] = v
    #     elif 'middle.2.weight' == k:
    #         new_pth['encoder.7.weight'] = v
    #     elif 'middle.2.bias' == k:
    #         new_pth['encoder.7.bias'] = v
    #     else:
    #         new_pth[k] = v
    # torch.save(new_pth, f'static/checkpoints/fuji/eterna-bleach-bypass/best-lab2.pth')
    # model.load_state_dict(new_pth)
    # print(list(model.state_dict().keys())[-1])
