import os

import torch
from skimage import data
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from loss import ChiSquareLoss

def image2hsit(img,show=False):
    # 读取RGB图像
    image = cv2.imread(img)  # 替换为你的图像文件路径
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 计算直方图
    r = torch.from_numpy(image_rgb).to(torch.float32).permute(2,1,0)
    a = torch.histc(input=r[0,:,:],bins=256,min=0,max=256)
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    # 将直方图归一化到0到1之间
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()
    # 将直方图堆叠成一个张量
    hist_tensor = np.vstack((hist_r, hist_g, hist_b))
    if show:
        # 可视化直方图
        plt.ylim(0,0.15)
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.show()
    return hist_tensor

def hist(img):
    plt.ylim(0, 0.15)
    img = cv2.imread(img)[:, :, ::-1]
    ar = img[:, :, 0].flatten()
    plt.hist(ar, bins=256, density=True, facecolor='r', edgecolor='r', stacked=True)
    ag = img[:, :, 1].flatten()
    plt.hist(ag, bins=256, density=True, facecolor='g', edgecolor='g', stacked=True)
    ab = img[:, :, 2].flatten()
    plt.hist(ab, bins=256, density=True, facecolor='b', edgecolor='b')
    plt.show()
    print(max(ar),max(ag),max(ab))


if __name__ == '__main__':
    a = image2hsit('/Users/maoyufeng/slash/project/filter-simulation/test/film_mask/18.jpg',show=True)
    # b = image2hsit('images/1696644460324322_mask.jpg',show=True)
    # # hist('/Users/maoyufeng/Downloads/浓郁色调/PA044733.jpg')
    # # hist('/Users/maoyufeng/Downloads/浓郁色调/PA044733_mask.jpg')
    # print(a.shape)
    # loss = ChiSquareLoss()
    # a = torch.from_numpy(a).permute(1,0)
    # b = torch.from_numpy(b).permute(1,0)
    # l = loss(a,b)
    # print(l)

    im = cv2.imread('/Users/maoyufeng/Downloads/test17.jpg')[:,:,::-1]
    cv2.imshow('test',im)
    cv2.waitKey(0)