import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


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
        #b,c,h,w = x.shape
        batch = x1.shape[0]
        loss = 0
        k_list =[]
        for x in [x1,x2]:
            mean_rgb = torch.mean(x,[2,3],keepdim=True)
            mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
            Drg = torch.pow(mr-mg,2)
            Drb = torch.pow(mr-mb,2)
            Dgb = torch.pow(mb-mg,2)
            k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + self.eps,0.5)
            k = k.reshape(batch)
            k_list.append(k)
        for i in range(0, batch):
            loss += torch.sum(torch.abs(k_list[0][i]-k_list[1][i]))
        loss /= batch
        return loss






if __name__ == '__main__':
    # img = transform(Image.open('/Users/maoyufeng/Downloads/iShot_2023-12-05_11.40.09.jpg'))
    # hist_channel = torch.histc(img[:, :, :] * 255, bins=256, min=0, max=255)
    # print(hist_channel)
    # hist, bin_edges = np.histogram(img[:, :, :] * 255, bins=256, range=(0,255))
    # print(torch.from_numpy(hist).float())
    # im1 = cv2.imread('/Users/maoyufeng/Library/Mobile Documents/com~apple~CloudDocs/实验/img/1_real.jpg')
    # im2 = cv2.imread('/Users/maoyufeng/Library/Mobile Documents/com~apple~CloudDocs/实验/img/1_rgb.jpg')
    # im1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0).to(torch.float32)/ 255.0
    # im2 = torch.from_numpy(im2).permute(2, 0, 1).unsqueeze(0).to(torch.float32)/ 255.0
    # # im1 = torch.rand((4,3,224,224))
    # # im2 = torch.rand((4,3,224,224))
    # loss = RGBLoss2()
    # print(loss(torch.cat([im1,im1,im1,im1]),torch.cat([im2,im2,im2,im2])))

    image1 = cv2.imread('/Users/maoyufeng/Downloads/filter-test/velvia/DSCF3575.jpg')  # 加载图像1
    image2 = cv2.imread('/Users/maoyufeng/Downloads/232133.jpg')  # 加载图像2
    image3 = cv2.imread('/Users/maoyufeng/Downloads/23213345.jpg')  # 加载图像2
    image1 = cv2.resize(image1,(image2.shape[1],image2.shape[0]))



    # 将图像转换为PyTorch张量，并确保它们的形状是相同的
    image1_tensor = torch.tensor(image1, dtype=torch.float32)
    image2_tensor = torch.tensor(image2, dtype=torch.float32)
    image3_tensor = torch.tensor(image3, dtype=torch.float32)
    # 获取图像的高度和宽度
    # height, width,_ = image1_tensor.shape
    # 计算每个像素上的L1 loss


    # losses = torch.sum(torch.abs(image1_tensor - image2_tensor))
    # print(losses)
    # losses = torch.sum(torch.abs(image1_tensor - image3_tensor))
    # print(losses)
    loss = F.l1_loss(image1_tensor, image2_tensor)
    print(f'Filter Small: {loss}')
    loss = F.l1_loss(image1_tensor, image3_tensor)
    print(f'Filter  Base: {loss}')

