from torch import nn
import numpy as np
import cv2
import torch

class FilterSimulation(nn.Module):
    """
    滤镜模拟
    Optimizer:AdamW
    lr:1e4
    Loss: RGBLoss+L1Loss
    Channel: RGB
    Epoch: 150
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
        x1 = self.encoder(x)    # 编码器
        x2 = self.decoder(x1)   # 解码器
        return x2


def image2block(image, patch_size=448, padding=16):
    patches, size_list = [], []
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
    H,W,C = image.shape
    # 从上到下 从左到右
    for x1 in range(padding, W - 2 * padding, patch_size):
        for y1 in range(padding, H - 2 * padding, patch_size):
            x2 = min(x1 + patch_size + padding, W)
            y2 = min(y1 + patch_size + padding, H)
            patch= image[y1 - padding:y2,x1 - padding:x2,:]
            size_list.append((x1 - padding, y1 - padding, patch.shape[1], patch.shape[0]))  # x,y,w,h
            # RGB Channel
            patch = torch.from_numpy(cv2.resize(patch, (patch_size, patch_size)) / 255.0)
            if len(patch.shape) == 2:
                patch = patch.unsqueeze(-1)
            patch = patch.permute(2, 0, 1).unsqueeze(0).float()
            patches.append(patch)
    return patches, size_list


def infer(image, model, device, patch_size=448, padding=16, batch=8):
    """
    image: 输入图片路径
    args: 各种参数
    """
    img = cv2.imread(image)  # 转RGB
    channel = model.state_dict()['decoder.4.bias'].shape[0]  # 获取加载的模型的通道
    if channel == 1:
        # 黑白滤镜
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 彩色滤镜
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 对每个小块进行推理
    target = np.zeros(shape=(img.shape),dtype=np.uint8)
    split_images, size_list = image2block(img, patch_size=patch_size, padding=padding)
    with torch.no_grad():
        for i in range(0, len(split_images), batch):
            input = torch.vstack(split_images[i:i + batch])
            input = input.to(device)
            output = model(input)
            for k in range(output.shape[0]):
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,0).detach().cpu().numpy()
                x, y, w, h = size_list[i + k]
                out = cv2.resize(out, (w, h))
                out = out[padding:h - padding, padding:w - padding]
                target[y:y+out.shape[0],x:x+out.shape[1],:]= out
    return cv2.cvtColor(target,cv2.COLOR_RGB2BGR)
