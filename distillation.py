import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 假设您已经有一个教师模型 FilmMaskSmall
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaskDataset
from loss import RGBLoss
from test import UNet,UNetStudent



seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


teacher_model = UNet()
teacher_model.load_state_dict(torch.load('/Users/maoyufeng/Downloads/polaroid/best.pth', map_location=device))
teacher_model = teacher_model.to(device=device)
teacher_model.eval()




# 定义蒸馏损失
class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.rgb_loss = RGBLoss()

    def forward(self, student_output, teacher_output, target):
        # 标准损失（例如 MSE）
        standard_loss = self.rgb_loss(student_output, target) + self.l1_loss(student_output,target)

        # 蒸馏损失（KL 散度）
        student_output_soft = nn.functional.log_softmax(student_output / self.temperature, dim=1)
        teacher_output_soft = nn.functional.softmax(teacher_output / self.temperature, dim=1)
        distillation_loss = self.kl_div_loss(student_output_soft, teacher_output_soft) * (self.temperature ** 2)
        # rgb_loss = self.rgb_loss()
        # 综合损失
        loss = self.alpha * standard_loss + (1 - self.alpha) * distillation_loss
        return loss


def train(data_path,training_channel,model_path,batch_size=8,lr=0.0001,temperature=3,epochs=100,alpha=0.5):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # 准备数据集和数据加载器
    train_data = MaskDataset(dataset_path=data_path, mode='train', channel=training_channel)
    val_data = MaskDataset(dataset_path=data_path, mode='val', channel=training_channel)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    # 创建学生模型和蒸馏损失
    student_model = UNetStudent()
    student_model = student_model.to(device)
    student_model.load_state_dict(torch.load('/Users/maoyufeng/slash/dataset/train_dataset/polaroid_dist/epoch0.pth', map_location=device))
    distillation_criterion = DistillationLoss(temperature=temperature,alpha=alpha)

    # 定义优化器
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    max_loss = 999.0
    # 训练学生模型
    for epoch in range(epochs):
        student_model.train()
        total_loss = []
        pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
        for org_img, target in train_loader:
            org_img = org_img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            # 获得教师模型的输出
            with torch.no_grad():
                teacher_outputs = teacher_model(org_img)
            # 计算学生模型的输出
            student_outputs = student_model(org_img)
            # 计算蒸馏损失
            loss = distillation_criterion(student_outputs, teacher_outputs, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            pbar.set_postfix(**{'loss': round(sum(total_loss)/len(total_loss), 4)})  # 参数列表
            pbar.update(1)  # 步进长度

        # 输出每个epoch的平均损失
        torch.save(student_model.state_dict(), os.path.join(model_path, f"epoch{epoch}.pth"))

        pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch + 1}: ")
        student_model.eval()
        val_total_loss = []
        with torch.no_grad():
            for org_img, target in val_loader:
                org_img = org_img.to(device)
                target = target.to(device)
                # 获得教师模型的输出
                teacher_outputs = teacher_model(org_img)
                # 计算学生模型的输出
                student_outputs = student_model(org_img)
                # 计算蒸馏损失
                loss = distillation_criterion(student_outputs, teacher_outputs, target)
                val_total_loss.append(loss.item())
                pbar.set_postfix(**{'loss': round(sum(val_total_loss) / len(val_loader), 4)})  # 参数列表
                pbar.update(1)  # 步进长度
        average_loss = sum(val_total_loss) / len(val_loader)
        if max_loss >= average_loss:
            max_loss = average_loss
            torch.save(student_model.state_dict(), os.path.join(model_path, "best.pth"))
        scheduler.step(average_loss)

if __name__ == '__main__':
    train(data_path='/Users/maoyufeng/slash/dataset/train_dataset/polaroid',
          training_channel='rgb',
          model_path='/Users/maoyufeng/slash/dataset/train_dataset/polaroid_dist',
          temperature=2,
          lr=1e-4,
          alpha=1)