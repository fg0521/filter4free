import json
import logging
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
from loss import ChiSquareLoss, HistogramLoss, EMDLoss
from models import FilmMask, FilterSimulation
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import Processor

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


class Trainer:
    def __init__(self, model, data_path, save_model_path,channel='rgb', pretrained_model_path=None, intensification=False, test_image=None):
        """
        model: 模型
        data_path: 训练数据
        save_model_path: 模型保存路径
        channel: 模型训练使用的通道 rgb/lab/gray
        pretrained_model_path: 预训练模型权重
        intensification: 数据增强
        test_image: 测试图像数据
        """
        
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.data_path = data_path
        self.channel = channel
        self.test_image = test_image
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
        if self.test_image is not None:
            self.test_path = os.path.dirname(test_image)
        # assert os.path.exists(data_path), 'Can Not Find Dataset For Training!'
        # if not os.path.exists(save_model_path):
        #     os.mkdir(save_model_path)
        self.save_model_path = save_model_path
        self.intensification = intensification
        # self.processor = Processor()

    def preprocess(self):
        # 进行图像增强操作
        if self.intensification:
            self.processor.run(input_path=self.data_path,
                               output_path=self.data_path)
        train_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='train', channel=self.channel)
        val_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='val', channel=self.channel)
        return train_data, val_data

    def train(self, epoch=200, lr=0.00001, batch_size=8, eval_step=5, early_stop_step=10, save_cfg=True):
        """
        epoch: 训练轮次
        lr: 学习率
        batch_size: 批次大小
        eval_step: 验证步长
        early_stop_step: 早停步长
        save_cfg: 记录config
        """
        train_data, val_data = self.preprocess()
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        self.model = self.model.to(self.device)
        # 定义损失函数和优化器
        mse_fn = nn.MSELoss()
        emd_fn = EMDLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)
        # 训练模型
        num_epochs = epoch
        max_loss = 1.0
        flag = 0
        training_loss,eval_loss = [],[]
        for epoch in range(num_epochs):
            loss_list = [[], []]
            pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}: ")
            epoch_loss = []
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
                pbar.set_postfix(**{'emd_loss': round(sum(loss_list[0]) / len(loss_list[0]), 4),
                                    'mse_loss': round(sum(loss_list[1]) / len(loss_list[1]), 4), })  # 参数列表
                pbar.update(1)  # 步进长度
                epoch_loss.append(loss1.item()+loss2.item())

            StepLR.step()
            torch.save(self.model.state_dict(), os.path.join(self.save_model_path, f"epoch{epoch}.pth"))
            if self.test_image is not None:
                self.infer(checkpoint=os.path.join(self.save_model_path, f"epoch{epoch}.pth"),
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
                eval_loss.append(avg_loss)
                if avg_loss <= max_loss:
                    max_loss = avg_loss
                    # 保存训练好的模型
                    torch.save(self.model.state_dict(), os.path.join(self.save_model_path, "best.pth"))
                    flag = 0
                else:
                    flag += 1
            if flag >= early_stop_step:
                logging.info(f'The Model Did Not Improve In {early_stop_step} Validations!')
                break
            training_loss.append(sum(epoch_loss)/len(epoch_loss))
        with open(os.path.join(self.save_model_path, 'train_loss.txt'), 'w') as file:
            [file.write(f"{line}\n") for line in training_loss]
        with open(os.path.join(self.save_model_path, 'eval_loss.txt'), 'w') as file2:
            [file2.write(f"{line}\n") for line in eval_loss]
        plt.plot(np.array(range(len(training_loss))), np.array(training_loss), c='r')  # 参数c为color简写，表示颜色,r为red即红色
        plt.plot(np.array(range(len(eval_loss))), np.array(eval_loss), c='b')  # 参数c为color简写，表示颜色,r为red即红色
        plt.legend(labels='train_loss')
        plt.savefig(os.path.join(self.save_model_path, 'train_loss.png'))
        if save_cfg:
            info = {
                'epoch': epoch,
                'learning_rate': lr,
                'batch_size': batch_size,
                'eval_step': eval_step,
                'device': str(self.device)
            }
            with open(os.path.join(self.save_model_path, 'config.json'), 'w') as f:
                json.dump(info, f, ensure_ascii=True)

    def infer(self, checkpoint, save_path, quality=100):
        self.model.load_state_dict(torch.load(checkpoint,map_location=self.device))
        model = self.model.to(self.device)
        if self.channel == 'rgb':
            input = transform(Image.open(self.test_image)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(input)
            out = Image.fromarray(
                torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        elif self.channel == 'lab':
            input = Image.open(self.test_image).resize((512, 512))
            input = np.array(input)
            input = np.clip((rgb2lab(input / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(self.device)
            with torch.no_grad():
                out = model(input)
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (lab2rgb(out * [100, 255, 255] - [0, 128, 128]) * 255).astype(np.uint8)
            out = Image.fromarray(out)
        elif self.channel == 'gray':
            input = transform(Image.open(self.test_image).convert('L')).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(input)
            out = Image.fromarray(
                torch.clamp(out.squeeze(0) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        out.save(save_path, quality=quality)


if __name__ == '__main__':
    trainer = Trainer(data_path='/Users/maoyufeng/slash/dataset/train_dataset/acros',
                      model=FilterSimulation(training=True,channel=1),
                      save_model_path='static/checkpoints/fuji/acros',
                      pretrained_model_path=None,
                      channel='gray')
    trainer.train(epoch=200, lr=0.002, batch_size=8, eval_step=5, early_stop_step=50)

