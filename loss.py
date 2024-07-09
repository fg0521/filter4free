import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        for x in range(15):  # 取VGG的前5层
            self.slice1.add_module(str(x), vgg[x])
        self.slice1.eval()  # 设置为评估模式
        for param in self.slice1.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.slice1(x)
        y_vgg = self.slice1(y)
        loss = F.mse_loss(x_vgg, y_vgg)  # 使用均方误差计算损失
        return loss

class ChiSquareLoss(nn.Module):
    """
    卡方分布损失
    """
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
    """
    直方图损失
    """
    def __init__(self, num_bins=256, margin=1):
        super(HistogramLoss, self).__init__()
        self.num_bins = num_bins
        self.margin = margin

    def forward(self, x, y):
        hist_x = self.compute_histogram(x)
        hist_y = self.compute_histogram(y)
        loss = self.histogram_loss(hist_x, hist_y)
        loss.requires_grad = True
        return loss

    def compute_histogram(self, img):
        hist = []
        for i in range(img.shape[0]):  # Iterate over channels
            try:
                hist_channel = torch.histc(img[i, :, :, :] * 255, bins=self.num_bins, min=0, max=255)
            except NotImplementedError:
                # not support for mps now, need to move tensor to cpu
                hist_channel = torch.histc(img.cpu()[i, :, :, :] * 255, bins=self.num_bins, min=0, max=255).to(
                    img.device)
            hist_channel = hist_channel / torch.sum(hist_channel)  # Normalize histogram
            hist.append(hist_channel)
        hist = torch.stack(hist, dim=1)
        return hist

    def histogram_loss(self, hist_x, hist_y):
        loss = torch.sum(torch.sqrt(hist_x) - torch.sqrt(hist_y)) ** 2
        loss = torch.clamp(loss, min=0.0, max=self.margin)
        return loss


class EMDLoss(nn.Module):
    """
    EMD损失
    """
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


class RGBLoss(nn.Module):
    """
    RGB Loss
    计算图片R、G、B三个通道两两之间的差异
    """
    def __init__(self):
        super(RGBLoss, self).__init__()
        self.eps = 1e-4

    def forward(self,x1,x2):
        rgb1 = torch.mean(x1,[2,3],keepdim=True)
        rgb2 = torch.mean(x2,[2,3],keepdim=True)
        Drg1,Drb1,Dgb1 = torch.pow(rgb1[:,0] - rgb1[:,1], 2),torch.pow(rgb1[:,1] - rgb1[:,2], 2),torch.pow(rgb1[:,2] - rgb1[:,0], 2)
        Drg2,Drb2,Dgb2 = torch.pow(rgb2[:,0] - rgb2[:,1], 2),torch.pow(rgb2[:,1] - rgb2[:,2], 2),torch.pow(rgb2[:,2] - rgb2[:,0], 2)
        k1 = torch.pow(torch.pow(Drg1, 2) + torch.pow(Drb1, 2) + torch.pow(Dgb1, 2) + self.eps, 0.5).squeeze()
        k2 = torch.pow(torch.pow(Drg2, 2) + torch.pow(Drb2, 2) + torch.pow(Dgb2, 2) + self.eps, 0.5).squeeze()
        loss = torch.mean(torch.abs(k1-k2))

        return loss



class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))  # train
        # samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 1)) # test

        return samplewise_emd.mean()


if __name__ == '__main__':
    torch.manual_seed(255)
    im1 = torch.rand((4,3,224,224))
    im2 = torch.rand((4,3,224,224))
    loss = RGBLoss()
    print(loss(im1,im2))


