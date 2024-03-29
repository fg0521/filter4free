import json
import logging
import os.path
import random
import glog as log
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MaskDataset, transform, Compose, ToTensor, RandomCropThreeInstances, \
    RandomHorizontalFlipThreeInstances
from torchvision import transforms
import torch.nn as nn

from infer import image2block
from loss import RGBLoss, EDMLoss,PerceptualLoss
from models import FilterSimulation,FilterSimulationFast, DNCM, Encoder
import numpy as np
import matplotlib.pyplot as plt
# from infer import image2block
import wandb

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #
wandb.login(key='78f2fd2cf17e26318b0724a2d317847d0bd74232')


class Trainer:
    def __init__(self, model, data_path, save_model_path, channel='rgb', resize=(700, 700), pretrained_model_path=None,
                 test_image=None):
        """
        model: 模型
        data_path: 训练数据
        save_model_path: 模型保存路径
        channel: 模型训练使用的通道 rgb/lab/gray
        pretrained_model_path: 预训练模型权重
        intensification: 数据增强
        test_image: 测试图像数据
        resize: 模型训练图像大小
        """
        self.wandb = wandb
        self.wandb.init(
            project='filter4free',
            resume=False,
            notes='./logs',
        )
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
        self.train_image_size = resize
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device), strict=False)
        if self.test_image is not None:
            self.test_path = os.path.dirname(test_image)
        assert os.path.exists(data_path), 'Can Not Find Dataset For Training!'
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        self.save_model_path = save_model_path



    def train(self, epoch_num=200, lr=0.00001, batch_size=8, eval_step=5, early_stop_step=10, save_cfg=True):
        """
        epoch: 训练轮次
        lr: 学习率
        batch_size: 批次大小
        eval_step: 验证步长
        early_stop_step: 早停步长
        save_cfg: 记录config
        ```经过测试 最适合的组合是 RGBLoss+L1Loss 在FilterSimulation(6个卷积)上进行训练 使用16batch 2e-4lr 可快速收敛```
        """
        train_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='train', channel=self.channel,
                                 resize=self.train_image_size)
        val_data = MaskDataset(dataset_path=os.path.join(self.data_path), mode='val', channel=self.channel,
                               resize=self.train_image_size)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        self.model = self.model.to(self.device)
        # 定义损失函数和优化器
        self.l1_fn = nn.L1Loss()
        self.rgb_fn = RGBLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        # 训练模型
        max_loss = 1.0
        flag = 0
        train_loss, eval_loss = [], []
        for epoch in range(1, epoch_num):
            self.model.train()
            loss_list = [[], []]
            pbar = tqdm(total=len(train_loader), desc=f"Epoch: {epoch}: ")
            epoch_train_loss = []
            for step,(org_img, goal_img) in enumerate(train_loader):
                org_img, goal_img = org_img.to(self.device), goal_img.to(self.device)
                optimizer.zero_grad()


                out = self.model(org_img)
                train_rgb_loss = self.rgb_fn(out, goal_img)
                train_l1_loss = self.l1_fn(out, goal_img)
                loss = train_rgb_loss + train_l1_loss
                loss.backward()
                optimizer.step()
                loss_list[0].append(train_rgb_loss.item())
                loss_list[1].append(train_l1_loss.item())
                pbar.set_postfix(**{'rgb_loss': round(sum(loss_list[0]) / len(loss_list[0]), 4),
                                    'l1_loss': round(sum(loss_list[1]) / len(loss_list[1]), 4), })  # 参数列表
                pbar.update(1)  # 步进长度
                epoch_train_loss.append(train_rgb_loss.item() + train_l1_loss.item())
                self.wandb.log({
                    "rgb_loss": train_rgb_loss.item(),
                    "l1_loss": train_l1_loss.item()
                }, commit=False)
                if step % 100 == 0 and step != 0:
                    self.visualize(org_im=org_img,true_im=goal_img,pred_im=out)

            StepLR.step()
            torch.save(self.model.state_dict(), os.path.join(self.save_model_path, f"epoch{epoch}.pth"))
            if self.test_image is not None:
                # 对测试图像进行推理
                self.test(checkpoint=os.path.join(self.save_model_path, f"epoch{epoch}.pth"),
                          save_path=os.path.join(self.test_path, f"epoch{epoch}.jpg"))

            if (epoch) % eval_step == 0:
                epoch_eval_loss = self.evaluation(val_loader=val_loader, epoch=epoch)
                eval_loss.append(epoch_eval_loss)
                if epoch_eval_loss <= max_loss:
                    max_loss = epoch_eval_loss
                    # 保存训练好的模型
                    torch.save(self.model.state_dict(), os.path.join(self.save_model_path, "best.pth"))
                    flag = 0
                else:
                    flag += 1
            if flag >= early_stop_step:
                logging.info(f'The Model Did Not Improve In {early_stop_step} Validations!')
                break
            train_loss.append(sum(epoch_train_loss) / len(epoch_train_loss))
        with open(os.path.join(self.save_model_path, 'train_loss.txt'), 'w') as file:
            [file.write(f"{line}\n") for line in train_loss]
        with open(os.path.join(self.save_model_path, 'eval_loss.txt'), 'w') as file2:
            [file2.write(f"{line}\n") for line in eval_loss]
        plt.plot(np.array(range(len(train_loss))), np.array(train_loss), c='r')  # 参数c为color简写，表示颜色,r为red即红色
        plt.plot(np.array(range(len(eval_loss))), np.array(eval_loss), c='b')  # 参数c为color简写，表示颜色,r为red即红色
        plt.legend(labels=['train_loss', 'eval_loss'])
        plt.savefig(os.path.join(self.save_model_path, 'loss.png'))
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

    def evaluation(self, val_loader, epoch):
        total_eval_loss = []
        self.model.eval()
        pbar = tqdm(total=len(val_loader), desc=f"Epoch: {epoch}: ")
        for val_org_img, val_goal_img in val_loader:
            val_org_img, val_goal_img = val_org_img.to(self.device), val_goal_img.to(self.device)
            out = self.model(val_org_img)
            val_rgb_loss = self.rgb_fn(out, val_goal_img)
            val_l1_loss = self.l1_fn(out, val_goal_img)
            val_loss = val_rgb_loss + val_l1_loss
            total_eval_loss.append(val_loss.item())
            pbar.set_postfix(**{'loss': round(sum(total_eval_loss) / len(total_eval_loss), 5)})  # 参数列表
            pbar.update(1)  # 步进长度
        avg_loss = sum(total_eval_loss) / len(val_loader)
        return avg_loss

    def test(self, checkpoint, save_path, quality=100, batch=8, padding=16):
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        test_img = Image.open(self.test_image)
        if self.channel == 3:
            target = Image.new('RGB', test_img.size)
        else:
            test_img = test_img.convert('L')
            target = Image.new('L', test_img.size)

        if test_img.width > self.train_image_size or test_img.height > self.train_image_size:
            split_images, size_list = image2block(test_img, patch_size=self.train_image_size, padding=padding)
        else:
            split_images, size_list = [test_img], [0, 0, test_img.width, test_img.height]
        with torch.no_grad():
            for i in tqdm(range(0, len(split_images), batch), desc='测试图像推理'):
                input = torch.vstack(split_images[i:i + batch])
                input = input.to(self.device)
                output = self.model(input)
                for k in range(output.shape[0]):
                    # RGB Channel
                    out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,
                                                                                               0).detach().cpu().numpy()
                    # # LAB Channel
                    # out = (lab2rgb(output[k, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * [100, 255, 255] - [0, 128,128]) * 255).astype(np.uint8)
                    x, y, w, h = size_list[i + k]
                    out = cv2.resize(out, (w, h))
                    if len(out.shape) == 3:
                        out = out[padding:h - padding, padding:w - padding, :]
                    else:
                        out = out[padding:h - padding, padding:w - padding]
                    target.paste(Image.fromarray(out), (x, y))
        target.save(save_path, quality=quality)


    def visualize(self, org_im,true_im,pred_im):
        idx = 0
        self.wandb.log({"examples": [
            self.wandb.Image(transforms.ToPILImage()(org_im[idx].cpu()), caption="Original"),
            self.wandb.Image(transforms.ToPILImage()(torch.clamp(true_im, min=0., max=1.)[idx].cpu()), caption="Truly"),
            self.wandb.Image(transforms.ToPILImage()(torch.clamp(pred_im, min=0., max=1.)[idx].cpu()), caption="Prediction"),
        ]}, commit=False)
        self.wandb.log({})





class Trainer2:
    def __init__(self):
        self._init_parameters()
        self.wandb = wandb
        self.wandb.init(
            project=self.PROJECT_NAME,
            resume=self.INIT_FROM is not None,
            notes=str(self.LOG_DIR),
        )

        self.dataset = MaskDataset(dataset_path=self.DATASET_ROOT,
                                   mode='train',
                                   channel='rgb', resize=self.IMG_SIZE)

        self.image_loader = DataLoader(dataset=self.dataset, batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)
        self.to_pil = transforms.ToPILImage()

        self.nDNCM = DNCM(self.k).to(torch.device('mps'))
        self.sDNCM = DNCM(self.k).to(torch.device('mps'))
        self.encoder = Encoder(self.sz, self.k).to(torch.device('mps'))

        # only learn parameters of sDNCM,nDNCM,D,S
        self.optimizer = torch.optim.Adam(list(self.sDNCM.parameters())
                                          + list(self.nDNCM.parameters())
                                          + list(self.encoder.D.parameters())
                                          + list(self.encoder.S.parameters()),
                                          lr=self.LR, betas=self.BETAS
                                          )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.SCHEDULER_STEP,
                                                         gamma=self.SCHEDULER_GAMMA)

        self.current_epoch = 0
        if self.INIT_FROM is not None and self.INIT_FROM != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints(self.INIT_FROM)

        # for param in self.nDNCM.parameters():
        #     param.requires_grad = False
        # for param in self.encoder.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.encoder.D.parameters():
        #     param.requires_grad = False
        # self.nDNCM.eval()
        # self.encoder.backbone.eval()
        # self.encoder.D.eval()
        # self.check_and_use_multi_gpu()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.RGBLoss = RGBLoss()
        # self.p_loss = PerceptualLoss()


    def _init_parameters(self):
        self.k = 16
        self.sz = 256
        self.LR = 3e-4
        self.BETAS = (0.9, 0.999)
        self.NUM_GPU = 0
        self.DATASET_ROOT = '/Users/maoyufeng/slash/dataset/train_dataset/gold200'
        self.IMG_SIZE = 256
        self.BATCH_SIZE = 24
        self.EPOCHS = 500
        self.LAMBDA = 10
        self.SCHEDULER_STEP = 24
        self.SCHEDULER_GAMMA = 0.1
        self.VISUALIZE_STEP = 50
        self.SHUFFLE = True
        self.CKPT_DIR = "./ckpts"
        self.INIT_FROM = ''
        self.PROJECT_NAME = "filter"
        self.LOG_DIR = "./logs"

    def run(self):
        max_loss = 999.0
        for e in range(self.EPOCHS):
            log.info(f"Epoch {e + 1}/{self.EPOCHS}")
            loss_list = []
            for step, (I, I_i) in enumerate(tqdm(self.image_loader, total=len(self.image_loader))):
                self.optimizer.zero_grad()
                I = I.to(torch.device('mps')).float()   # original image
                I_i = I_i.to(torch.device('mps')).float()   # filter image
                d, r = self.encoder(I)
                d_i, r_i = self.encoder(I_i)
                Z = self.nDNCM(I, d)  # content of original image
                Z_i = self.nDNCM(I_i, d_i)  # content of filter image

                # r_i = torch.vstack([torch.mean(r_i, dim=0) for i in range(r_i.shape[0])])   # get the average feature of a batch images
                Y = self.sDNCM(Z_i, r)
                Y_i = self.sDNCM(Z, r_i)  # predict the Y_i by adding OriginalImageContent and FilterImageColor

                consistency_loss = self.l2_loss(Z, Z_i)
                reconstruction_loss = self.l1_loss(Y_i, I_i) + self.l1_loss(Y,I)
                final_loss = reconstruction_loss + self.LAMBDA * consistency_loss
                # reconstruction_loss.backward()
                final_loss.backward()
                self.optimizer.step()
                self.wandb.log({
                    "consistency_loss": consistency_loss.item(),
                    "reconstruction_loss": reconstruction_loss.item()
                }, commit=False)
                if step % self.VISUALIZE_STEP == 0 and step != 0:
                    self.visualize(I, I_i, Y,Y_i,Z,Z_i)
                else:
                    self.wandb.log({})
                step_loss = reconstruction_loss.item()
                loss_list.append(step_loss)

            self.scheduler.step()
            each_loss = sum(loss_list) / len(loss_list)
            if each_loss < max_loss:
                max_loss = each_loss
                self.do_checkpoint()
                # torch.save(r_i, f"{self.CKPT_DIR}/r_i_ckpt.pth")

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.NUM_GPU > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs...")
            self.nDNCM = torch.nn.DataParallel(self.nDNCM)
            self.sDNCM = torch.nn.DataParallel(self.sDNCM)
            self.encoder = torch.nn.DataParallel(self.encoder)
        else:
            log.info(f"GPU ID: {torch.cuda.current_device()}")
            self.nDNCM = self.nDNCM
            self.sDNCM = self.sDNCM
            self.encoder = self.encoder

    def do_checkpoint(self):
        os.makedirs(str(self.CKPT_DIR), exist_ok=True)
        checkpoint = {
            # 'epoch': self.current_epoch,
            'nDCNM': self.nDNCM.module.state_dict() if isinstance(self.nDNCM,
                                                                  torch.nn.DataParallel) else self.nDNCM.state_dict(),
            'sDCNM': self.sDNCM.module.state_dict() if isinstance(self.sDNCM,
                                                                  torch.nn.DataParallel) else self.sDNCM.state_dict(),
            'encoder': self.encoder.module.state_dict() if isinstance(self.encoder,
                                                                       torch.nn.DataParallel) else self.encoder.state_dict(),

            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, f"{self.CKPT_DIR}/latest_ckpt.pth")

    def load_checkpoints(self, ckpt_path):
        checkpoints = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.nDNCM.load_state_dict(checkpoints["nDCNM"])
        self.sDNCM.load_state_dict(checkpoints["sDCNM"])
        self.encoder.load_state_dict(checkpoints["encoder"])
        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v

    def visualize(self,I, I_i, Y,Y_i,Z,Z_i):
        idx = 0
        self.wandb.log({"examples": [
            self.wandb.Image(self.to_pil(I[idx].cpu()), caption="I"),
            self.wandb.Image(self.to_pil(I_i[idx].cpu()), caption="I_i"),
            self.wandb.Image(self.to_pil(torch.clamp(Y[idx].cpu(), min=0., max=1.)), caption="Y"),
            self.wandb.Image(self.to_pil(torch.clamp(Y_i[idx].cpu(), min=0., max=1.)), caption="Y_i"),
            self.wandb.Image(self.to_pil(torch.clamp(Z[idx].cpu(), min=0., max=1.)), caption="Z"),
            self.wandb.Image(self.to_pil(torch.clamp(Z_i[idx].cpu(), min=0., max=1.)), caption="Z_i"),
        ]}, commit=False)
        self.wandb.log({})


if __name__ == '__main__':
    trainer = Trainer(data_path='/Users/maoyufeng/slash/dataset/train_dataset/portra160nc',
                      model=FilterSimulation(training=True),
                      save_model_path='static/checkpoints/kodak/portra160nc',
                      pretrained_model_path='static/checkpoints/kodak/portra160nc/best-ConvTranspose2d.pth',
                      channel='rgb')
    trainer.train(epoch_num=500, lr=2e-5, batch_size=8, eval_step=1, early_stop_step=30)

    # trainer = Trainer2()
    # trainer.run()
