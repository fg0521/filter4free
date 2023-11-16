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
from models import FilterSimulation



seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


if torch.cuda.is_available():
    device = torch.device('cuda:0')
# elif torch.backends.mps.is_built():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')


teacher_model = FilterSimulation()
teacher_model.load_state_dict(torch.load('static/checkpoints/olympus/best.pth', map_location=device))
teacher_model = teacher_model.to(device=device)
teacher_model.eval()


# 定义一个更小的学生模型
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x2 = self.decoder(self.encoder(x))
        return x2


# 定义蒸馏损失
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        student_probs = torch.softmax(student_output / self.temperature, dim=1)
        teacher_probs = torch.softmax(teacher_output / self.temperature, dim=1)
        loss1 = nn.KLDivLoss(reduction = 'batchmean')(torch.log(student_probs), teacher_probs)
        loss2 = nn.MSELoss()(torch.log(student_probs),teacher_probs)
        loss = loss1+loss2
        print(loss1.item(),loss2.item())
        return loss

def train(data_path,training_channel,model_path,batch_size=8,lr=1e-4,temperature=3,epochs=100):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # 准备数据集和数据加载器
    train_data = MaskDataset(dataset_path=data_path, mode='train', channel=training_channel)
    val_data = MaskDataset(dataset_path=data_path, mode='val', channel=training_channel)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    # 创建学生模型和蒸馏损失
    student_model = StudentModel()
    student_model = student_model.to(device)
    student_model.load_state_dict(torch.load('static/checkpoints/olympus_dist/best.pth', map_location=device))
    distillation_criterion = DistillationLoss(temperature=temperature)

    # 定义优化器
    optimizer = optim.AdamW(student_model.parameters(), lr=lr)
    max_loss = 999.0
    # 训练学生模型
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
        for org_img, _ in train_loader:
            org_img = org_img.to(device)
            optimizer.zero_grad()
            # 获得教师模型的输出
            with torch.no_grad():
                teacher_outputs = teacher_model(org_img)
            # 计算学生模型的输出
            student_outputs = student_model(org_img)
            # 计算蒸馏损失
            loss = distillation_criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(**{'loss': round(loss.item(), 6)})  # 参数列表
            pbar.update(1)  # 步进长度

        # 输出每个epoch的平均损失
        torch.save(student_model.state_dict(), os.path.join(model_path, f"epoch{epoch}.pth"))

        if (epoch + 1) % 10 == 0:
            student_model.eval()
            val_total_loss = 0.0
            for org_img, _ in val_loader:
                org_img = org_img.to(device)
                optimizer.zero_grad()
                # 获得教师模型的输出
                with torch.no_grad():
                    teacher_outputs = teacher_model(org_img)
                # 计算学生模型的输出
                student_outputs = student_model(org_img)
                # 计算蒸馏损失
                loss = distillation_criterion(student_outputs, teacher_outputs)
                loss.backward()
                optimizer.step()
                val_total_loss += loss.item()
                pbar.set_postfix(**{'loss': round(loss.item(), 6)})  # 参数列表
                pbar.update(1)  # 步进长度
            average_loss = val_total_loss / len(val_loader)
            if max_loss>=average_loss:
                max_loss = average_loss
                torch.save(student_model.state_dict(), os.path.join(model_path, "best.pth"))


if __name__ == '__main__':
    train(data_path='/Users/maoyufeng/slash/dataset/olympus浓郁色彩',
          training_channel='rgb',
          model_path='static/checkpoints/olympus_dist')