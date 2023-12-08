import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from math import exp

from dataset import transform


class ChiSquareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = 1e-10
        self.bins = 256

    def forward(self, hist1, hist2):
        # batch_size,channel,W,H
        assert hist1.size() == hist2.size(), "Histograms must have the same shape"
        total_loss = 0.
        batch_size = hist1.size()[0]
        hist1 = hist1 * 255
        hist2 = hist2 * 255
        for i in range(batch_size):
            r1 = torch.histc(hist1[i, 0, :, :], bins=self.bins, min=0, max=256)
            g1 = torch.histc(hist1[i, 1, :, :], bins=self.bins, min=0, max=256)
            b1 = torch.histc(hist1[i, 2, :, :], bins=self.bins, min=0, max=256)
            h1 = torch.cat((r1, g1, b1), dim=0).unsqueeze(0)
            h1 = h1 / h1.sum()

            # print(h1.shape)
            r2 = torch.histc(hist2[i, 0, :, :], bins=self.bins, min=0, max=256)
            g2 = torch.histc(hist2[i, 1, :, :], bins=self.bins, min=0, max=256)
            b2 = torch.histc(hist2[i, 2, :, :], bins=self.bins, min=0, max=256)
            h2 = torch.cat((r2, g2, b2), dim=0).unsqueeze(0)
            h2 = h2 / h2.sum()

            # print(h2.shape)
            chi_squared = torch.sum((h1 - h2) ** 2 / (h1 + h2 + self.bias))
            total_loss = total_loss + chi_squared
            # You can optionally normalize the chi_squared value by the number of bins.
        chi_squared = total_loss / batch_size
        chi_squared.requires_grad = True
        return chi_squared


class HistogramLoss(nn.Module):
    def __init__(self, num_bins=256, margin=1):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins
        self.margin = margin

    def forward(self, x, y):
        hist_x = self.compute_histogram(x)
        hist_y = self.compute_histogram(y)
        loss = self.histogram_loss(hist_x, hist_y)

        return loss

    def compute_histogram(self, img):
        hist = []
        for i in range(img.shape[0]):  # Iterate over channels
            hist_channel = torch.histc(img[i, :, :, :] * 255, bins=self.num_bins, min=0, max=255)
            hist_channel = hist_channel / torch.sum(hist_channel)  # Normalize histogram
            hist.append(hist_channel)
        hist = torch.stack(hist, dim=1)
        return hist

    def histogram_loss(self, hist_x, hist_y):
        loss = torch.sum(torch.sqrt(hist_x) - torch.sqrt(hist_y)) ** 2
        loss = torch.clamp(loss, min=0.0, max=self.margin)
        return loss


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
        self.num_bins = 256
        self.scalar = 1

    def compute_histogram(self, img):
        hist = []
        for i in range(img.shape[0]):  # Iterate over channels
            try:
                hist_channel = torch.histc(img[i, :, :, :] * 255, bins=self.num_bins, min=0, max=255)
            except NotImplementedError:
                # not support for mps now, need to move tensor to cpu
                hist_channel = torch.histc(img.cpu()[i, :, :, :] * 255, bins=self.num_bins, min=0, max=255).to(img.device)
            hist_channel = hist_channel / torch.sum(hist_channel)  # Normalize histogram
            hist.append(hist_channel)
        hist = torch.stack(hist, dim=1)
        return hist

    def forward(self,im1, im2):
        hist_dist1 = self.compute_histogram(im1)
        hist_dist2 = self.compute_histogram(im2)
        # 计算两个分布的累积分布函数（CDF）
        try:
            hist_dist1_cumsum = torch.cumsum(hist_dist1, dim=0)
            hist_dist2_cumsum = torch.cumsum(hist_dist2, dim=0)
        except NotImplementedError:
            # not support for mps, need to move tensor to cpu
            hist_dist1_cumsum = torch.cumsum(hist_dist1.cpu(), dim=0).to(hist_dist1.device)
            hist_dist2_cumsum = torch.cumsum(hist_dist2.cpu(), dim=0).to(hist_dist2.device)
        # 计算EMD
        emd_loss = torch.norm(hist_dist1_cumsum - hist_dist2_cumsum, p=1,dim=0)
        # 对每个批次每个通道求平均损失
        total_loss = torch.sum(emd_loss)/hist_dist2.shape[0]*self.scalar/3

        # 在channel、w和h上求和，得到每个批次的MSE损失
        # mse_loss = torch.mean((im1 - im2) ** 2, dim=(1, 2, 3))
        # total_loss = mse_loss+emd_loss
        total_loss.requires_grad = True
        return total_loss



# 5.SSIM loss
# 生成一位高斯权重，并将其归一化
def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss/torch.sum(gauss)  # 归一化


# x=gaussian(3,1.5)
# # print(x)
# x=x.unsqueeze(1)
# print(x.shape) #torch.Size([3,1])
# print(x.t().unsqueeze(0).unsqueeze(0).shape) # torch.Size([1,1,1, 3])

# 生成滑动窗口权重，创建高斯核：通过一维高斯向量进行矩阵乘法得到
def create_window(window_size,channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size,1
    # mm:矩阵乘法 t:转置矩阵 ->1,1,window_size,_window_size
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # expand:扩大张量的尺寸，比如3,1->3,4则意味将输入张量的列复制四份，
    # 1,1,window_size,_window_size->channel,1,window_size,_window_size
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 构造损失函数用于网络训练或者普通计算SSIM值
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# 普通计算SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)




if __name__ == '__main__':
    img = transform(Image.open('/Users/maoyufeng/Downloads/iShot_2023-12-05_11.40.09.jpg'))
    hist_channel = torch.histc(img[:, :, :] * 255, bins=256, min=0, max=255)
    print(hist_channel)
    hist, bin_edges = np.histogram(img[:, :, :] * 255, bins=256, range=(0,255))
    print(torch.from_numpy(hist).float())
