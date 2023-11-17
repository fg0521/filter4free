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
from models import FilmMask,FilterSimulation
import cv2
import numpy as np

from preprocessing import Processor

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #



class Trainer():
    def __init__(self,model,data_path,model_path,intensification=True,test_image=None):
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.data_path = data_path
        self.train_channel = 'rgb'
        self.test_image = test_image
        if self.test_image is not None:
            self.test_path = os.path.dirname(test_image)
        assert os.path.exists(data_path), 'Can Not Find Dataset For Training!'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model_path = model_path
        self.intensification = intensification
        self.processor = Processor()


    def preprocess(self):
        # 进行图像增强操作
        if self.intensification:
            self.processor.run(input_path=self.data_path,
                          output_path=self.data_path)
        train_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='train', channel=self.train_channel)
        val_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='val', channel=self.train_channel)
        return train_data,val_data

    def train(self,epoch=100,lr=1e-3,batch_size=8,eval_step=10):
        """
        epoch: 训练轮次
        lr: 学习率
        batch_size: 批次大小
        """

        train_data,val_data = self.preprocess()
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        self.model = self.model.to(self.device)
        torch.load(strict=False)
        # 定义损失函数和优化器
        mse_fn = nn.MSELoss()
        emd_fn = EMDLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)
        # 训练模型
        num_epochs = epoch
        max_loss = 1.0
        for epoch in range(num_epochs):
            loss_list = [[], []]
            pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
            for org_img, goal_img in train_loader:
                org_img, goal_img = org_img.to(self.device), goal_img.to(self.device)
                optimizer.zero_grad()
                out = self.model(org_img)
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
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f"epoch{epoch}.pth"))
            if self.test_image is not None:
                self.infer(checkpoint=os.path.join(self.model_path, f"epoch{epoch}.pth"),
                      save_path=os.path.join(self.test_path, f"epoch{epoch}.jpg"))

            if (epoch + 1) % eval_step == 0:
                valid_epoch_loss = []
                self.model.eval()
                pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch + 1}: ")
                for val_org_img, val_goal_img in val_loader:
                    val_org_img, val_goal_img = val_org_img.to(self.device), val_goal_img.to(self.device)
                    out = self.model(val_org_img)
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
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, "best.pth"))

    def infer(self,checkpoint, save_path,quality=100):
        self.model.load_state_dict(torch.load(checkpoint))
        model = self.model.to(self.device)
        if self.train_channel == 'rgb':
            input = transform(Image.open(self.test_image)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(input)
            out = Image.fromarray(torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        else:
            input = Image.open(self.test_image).resize((512, 512))
            input = np.array(input)
            input = np.clip((rgb2lab(input / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(self.device)
            with torch.no_grad():
                out = model(input)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (lab2rgb(out * [100, 255, 255] - [0, 128, 128]) * 255).astype(np.uint8)
            out = Image.fromarray(out)
        out.save(save_path,quality=quality)


if __name__ == '__main__':
    trainer =Trainer(data_path='/Users/maoyufeng/Downloads/1',
                     model=FilterSimulation(),
                     model_path='static/checkpoints/fuji/provia')
    trainer.train()
