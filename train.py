import os.path
import random

import skimage
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MaskDataset, transform
import torch.nn as nn
from loss import ChiSquareLoss, ColorTransferLoss,HistogramLoss
from models import Olympus, Unet, RGBUnet,FilmMask,FilmMask2
import cv2
import numpy as np
import matplotlib.pyplot as plt

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


model_zoo ={
    'olympus':Olympus(),
    'film_mask':FilmMask2(),
    'unet':Unet(),
    'rgb':RGBUnet(),
}

def train(data_path, save_path):
    train_data = MaskDataset(dataset_path=data_path, mode='train', channel='lab')
    val_data = MaskDataset(dataset_path=data_path, mode='val', channel='lab')
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=8, shuffle=True)
    # model = Olympus()
    model = Unet()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    # device = torch.device('cuda:0')
    # model.load_state_dict(torch.load('checkpoints/olympus/epoch0.pth',map_location=device))
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)
    # 训练模型
    num_epochs = 300
    max_loss = 1.0
    # lr_list = []
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(mask)
            loss = criterion(out, img)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'loss': round(loss.item(), 5)})  # 参数列表
            pbar.update(1)  # 步进长度
        StepLR.step()
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch}.pth"))
        infer(epoch=epoch, save_path=os.path.join(save_path, f"epoch{epoch}.pth"))
        if (epoch + 1) % 3 == 1:
            valid_epoch_loss = []
            model.eval()
            pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch + 1}: ")
            for val_img, val_mask in val_loader:
                val_img, val_mask = val_img.to(device), val_mask.to(device)
                out = model(val_mask)
                val_loss = criterion(out, val_img)  # 假设你有与图像相应的目标色彩分布 自定义函数，获取目标色彩分布
                valid_epoch_loss.append(val_loss.item())
                pbar.set_postfix(**{'loss': round(val_loss.item(), 5)})  # 参数列表
                pbar.update(1)  # 步进长度
            avg_loss = sum(valid_epoch_loss) / len(val_loader)
            if avg_loss <= max_loss:
                max_loss = avg_loss
                # 保存训练好的模型
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))


def train_with_hist(model_name,data_path, save_path):
    train_data = MaskDataset(dataset_path=data_path, mode='train', channel='rgb')
    val_data = MaskDataset(dataset_path=data_path, mode='val', channel='rgb')
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=8, shuffle=True)
    model = model_zoo[model_name]
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    # elif torch.backends.mps.is_built():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)
    # 定义损失函数和优化器
    criterion1 = nn.MSELoss()
    criterion = ChiSquareLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)
    # 训练模型
    num_epochs = 300
    max_loss = 1.0
    # lr_list = []
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(mask)
            loss1 = criterion(out, img)
            loss2 = criterion1(out, img)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**{'mse_loss': round(loss2.item(), 4),'hist_loss':round(loss1.item(),4)})  # 参数列表
            pbar.update(1)  # 步进长度
        StepLR.step()
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch{epoch}.pth"))
        infer_with_hist(model_name,epoch=epoch, save_path=os.path.join(save_path, f"epoch{epoch}.pth"))
        if (epoch + 1) % 10 == 0:
            valid_epoch_loss = []
            model.eval()
            pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch + 1}: ")
            for val_img, val_mask in val_loader:
                val_img, val_mask = val_img.to(device), val_mask.to(device)
                out = model(val_mask)
                val_loss1 = criterion(out, val_img)
                val_loss2 = criterion1(out, val_img)
                val_loss = val_loss1 + val_loss2
                valid_epoch_loss.append(val_loss.item())
                pbar.set_postfix(**{'loss': round(val_loss.item(), 5)})  # 参数列表
                pbar.update(1)  # 步进长度
            avg_loss = sum(valid_epoch_loss) / len(val_loader)
            if avg_loss <= max_loss:
                max_loss = avg_loss
                # 保存训练好的模型
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))


def infer(epoch, save_path):
    # model =Olympus()
    model = Unet()
    model.load_state_dict(torch.load(save_path))
    model.to('cpu')
    img = cv2.imread('images/1696644460324322_mask.jpg')[:, :, ::-1]
    H, W, C = img.shape
    input = cv2.resize(img, (512, 512))
    input = torch.from_numpy(skimage.color.rgb2lab(input) / 128).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        out = model(input)
    out = out.squeeze(0).permute(1, 2, 0).numpy()
    out = skimage.color.lab2rgb(out) * 128
    out = cv2.resize(out, (W, H)) * 255
    out = out[:, :, ::-1]
    cv2.imwrite(f'test/film_mask/{epoch}.jpg', out)


def infer_with_hist(model_name,epoch, save_path):
    model = model_zoo[model_name]
    model.load_state_dict(torch.load(save_path))
    model.to(torch.device('cpu'))
    img_tensor = transform(Image.open('images/1696644460324322_mask.jpg'))
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
    image = Image.fromarray(torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    image.save(f'test/film_mask/{epoch}.jpg')


if __name__ == '__main__':
    # train(data_path='/Users/maoyufeng/Downloads/Pytorch-UNet/data3',
    #       save_path='checkpoints/film_mask')

    train_with_hist(model_name='film_mask',data_path='/Users/maoyufeng/Downloads/main',
                    save_path='checkpoints/film_mask')
