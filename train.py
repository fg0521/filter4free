import os.path
import random

import skimage
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MaskDataset, transform
import torch.nn as nn
from loss import ChiSquareLoss, HistogramLoss,EMDLoss
from models import Olympus, FilmMask
import cv2
import numpy as np

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #

model_zoo = {
    'olympus': Olympus(),
    'film_mask': FilmMask(),
}
if torch.cuda.is_available():
    device = torch.device('cuda:0')
# elif torch.backends.mps.is_built():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')

def train(model_name,
          data_path,
          model_path,
          test_image,
          test_path,
          training_channel='rgb',
          epoch=100,
          lr=1e-3,
          batch_size=8,
          ):
    """
    model_name: 模型名称
    data_path: 训练数据路径
    model_path: 模型保存路径
    test_image: 测试图片路径，每训练1个epoch将对图片进行测试
    test_path: 测试图片结果保存路径
    training_channel: 训练通道 采用rgb或者lab通道
    epoch: 训练轮次
    lr: 学习率
    batch_size: 批次大小
    """
    assert os.path.exists(data_path),'Can Not Find Dataset For Training!'
    assert os.path.exists(test_image),'Can Not Find Image For Testing!'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    train_data = MaskDataset(dataset_path=data_path, mode='train', channel=training_channel)
    val_data = MaskDataset(dataset_path=data_path, mode='val', channel=training_channel)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    model = model_zoo[model_name]
    model = model.to(device)
    # 定义损失函数和优化器
    mse_fn = nn.MSELoss()
    emd_fn = EMDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)
    # 训练模型
    num_epochs = epoch
    max_loss = 1.0
    for epoch in range(num_epochs):
        loss_list = [[], []]
        pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
        for org_img, goal_img in train_loader:
            org_img, goal_img = org_img.to(device), goal_img.to(device)
            optimizer.zero_grad()
            out = model(org_img)
            loss1 = emd_fn(out, goal_img)
            loss2 = mse_fn(out, goal_img)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            loss_list[0].append(loss1.item())
            loss_list[1].append(loss2.item())
            pbar.set_postfix(**{'hist_loss': round(sum(loss_list[0])/len(loss_list[0]), 4),
                                'mse_loss': round(sum(loss_list[1])/len(loss_list[1]), 4), })  # 参数列表
            pbar.update(1)  # 步进长度
        StepLR.step()
        torch.save(model.state_dict(), os.path.join(model_path, f"epoch{epoch}.pth"))

        infer(model_name,
              model_path=os.path.join(model_path, f"epoch{epoch}.pth"),
              test_image=test_image,
              save_path=os.path.join(test_path, f"epoch{epoch}.jpg"),
              mode=training_channel)

        if (epoch + 1) % 10 == 0:
            valid_epoch_loss = []
            model.eval()
            pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch + 1}: ")
            for val_org_img, val_goal_img in val_loader:
                val_org_img, val_goal_img = val_org_img.to(device), val_goal_img.to(device)
                out = model(val_org_img)
                val_loss1 = emd_fn(out, val_goal_img)
                val_loss2 = mse_fn(out, val_goal_img)
                val_loss = val_loss1 + val_loss2
                valid_epoch_loss.append(val_loss.item())
                pbar.set_postfix(**{'loss': round(val_loss.item(), 5)})  # 参数列表
                pbar.update(1)  # 步进长度
            avg_loss = sum(valid_epoch_loss) / len(val_loader)
            if avg_loss <= max_loss:
                max_loss = avg_loss
                # 保存训练好的模型
                torch.save(model.state_dict(), os.path.join(model_path, "best.pth"))


def infer(model_name, model_path, test_image, save_path, mode='rgb'):
    model = model_zoo[model_name]
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    if mode == 'rgb':
        input = transform(Image.open(test_image)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input)
        out = Image.fromarray(torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    else:
        input = Image.open(test_image).resize((512, 512))
        input = np.array(input)
        input = np.clip((rgb2lab(input / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
        input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(device)
        with torch.no_grad():
            out = model(input)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (lab2rgb(out * [100, 255, 255] - [0, 128, 128]) * 255).astype(np.uint8)
        out = Image.fromarray(out)
    out.save(save_path)


if __name__ == '__main__':
    # train(model_name='olympus',
    #       data_path='/Users/maoyufeng/slash/dataset/olympus浓郁色彩',
    #       model_path='checkpoints/olympus',
    #       test_image='images/1696644460324322_mask.jpg',
    #       test_path='test/olympus',
    #       training_channel='rgb')

    train(model_name='film_mask',
          data_path='/Users/maoyufeng/slash/dataset/color_mask',
          model_path='checkpoints/film_mask2',
          test_image='images/mask.jpg',
          test_path='test/film_mask2',
          training_channel='rgb')
