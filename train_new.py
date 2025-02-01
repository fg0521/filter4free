"""

neural preset with cnn
step1: pretrain on MSCOCO Dataset
step2: continue train on Personal Dataset

model is on pretrained_model dir!!!
"""
import itertools
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance
from pillow_lut import load_cube_file
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import wandb
from torchvision import models, transforms
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
from tqdm import tqdm

from loss import RGBLoss
from models import UCM, Encoder, Shader

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


class MSCOCODataset(Dataset):
    def __init__(self, img_list, lut_path='/data/datasets/LUTS/'):
        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.luts_path = lut_path
        self.luts = [os.path.join(self.luts_path, i) for i in os.listdir(self.luts_path) if i.endswith('.cube')]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img1_path = self.img_list[idx]
        img1 = Image.open(img1_path).convert(mode='RGB')
        lut1 = load_cube_file(random.choice(self.luts))
        lut2 = load_cube_file(random.choice(self.luts))
        img2 = img1.filter(lut1)
        img1 = img1.filter(lut2)
        img2 = self.transform(img2)
        img1 = self.transform(img1)
        return img1, img2


class ColorTransferDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.luts_path = '/data/datasets/LUTS/'
        # self.luts = [i for i in os.listdir(self.luts_path) if i.endswith('.cube')]
        self.luts = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if len(self.img_list[idx]) == 2:
            # [(org,filter)]
            img1, img2 = self.img_list[idx]
            img1 = Image.open(img1).convert(mode='RGB')
            img2 = Image.open(img2).convert(mode='RGB')
            # im1 = img1.resize((random.randint(200, 600), random.randint(200, 600)))
            # im2 = img2.resize((random.randint(200, 600), random.randint(200, 600)))
            # img1_random = self.transform2(im1)
            # img2_random = self.transform2(im2)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        else:
            img1_path = self.img_list[idx]
            img1 = Image.open(img1_path).convert(mode='RGB')
            lut1 = load_cube_file(random.choice(self.luts))
            lut2 = load_cube_file(random.choice(self.luts))
            img2 = img1.filter(lut1)
            img1 = img1.filter(lut2)
            img2 = self.transform(img2)
            img1 = self.transform(img1)

        return img1, img2


def lr_lambda(epoch):
    return 1 if epoch < 24 else 0.1


def train(train_dataloader, val_dataloader, num_epochs, learning_rate,
          val_epoch=3, save_path='./checkpoint', filter_num=-1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    optimizer = optim.Adam(params=itertools.chain(encoder.parameters(),sNet.parameters(),cNet.parameters()),
                           lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda)

    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    # rgb_loss = RGBLoss()
    step = 0
    for epoch in range(num_epochs):
        encoder.train()
        sNet.train()
        cNet.train()
        for img1, img2 in tqdm(train_dataloader, desc=f"Epoch:{epoch + 1}"):
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()

            content_feat1, color_feat1 = encoder(img1)
            content_feat2, color_feat2 = encoder(img2)

            # 训练一种特定的lut，先按照原来的训练方式训练，经过若干个epoch后将色彩特征进行均值处理
            if filter_num == 1:
                bs = img1.shape[0]
                color_feat1 = color_feat1.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)
                color_feat2 = color_feat2.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)

            content1 = sNet(img1, content_feat1)
            content2 = sNet(img2, content_feat2)

            color1 = cNet(content2, color_feat1)
            color2 = cNet(content1, color_feat2)

            consistency_loss = l2_loss(content1, content2)
            reconstruction_loss = l1_loss(color1, img1) + l1_loss(color2, img2)
            # color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)

            msg = {
                "reconstruction_loss": reconstruction_loss.item(),
                # 'rgb_loss': color_loss.item(),
                "consistency_loss": consistency_loss.item(),
            }
            step += 1
            if step % 100 == 0:
                visualize(img1, img2, color1, color2, content1, content2)
            else:
                wandb.log({})
            loss = reconstruction_loss + 10 * consistency_loss
            wandb.log(msg, commit=False)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (epoch + 1) % val_epoch == 0:
            if filter_num==1:
                val(val_dataloader=val_dataloader,
                save_path=save_path, filter_num=filter_num,
                filter_color=color_feat1[0:1, :, :, :])
            else:
                val(val_dataloader=val_dataloader,
                    save_path=save_path, filter_num=filter_num,
                    filter_color=None)

        checkpoints = {"encoder": encoder.state_dict(),
                       'sNet': sNet.state_dict(),
                       'cNet': cNet.state_dict(),
                       }

        if filter_num == 1:
            checkpoints.update({"filter_color": color_feat1[0:1, :, :, :],
                                "org_color": color_feat2[0:1, :, :, :]})
        torch.save(checkpoints, os.path.join(save_path, f'train_{epoch + 1}.pth'))


def val(val_dataloader, save_path, filter_num,filter_color=None):
    encoder.eval()
    sNet.eval()
    cNet.eval()
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    # rgb_loss = RGBLoss()
    val_loss = []
    for img1, img2 in tqdm(val_dataloader):
        img1, img2 = img1.to(device), img2.to(device)
        with torch.no_grad():

            content_feat1, color_feat1 = encoder(img1)
            content_feat2, color_feat2 = encoder(img2)

            # 训练一种特定的lut
            if filter_num == 1:
                bs = img1.shape[0]
                color_feat1 = color_feat1.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)
                color_feat2 = color_feat2.mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1)

            content1 = sNet(img1, content_feat1)
            content2 = sNet(img2, content_feat2)

            color1 = cNet(content2, color_feat1)
            color2 = cNet(content1, color_feat2)

            consistency_loss = l2_loss(content1, content2)
            reconstruction_loss = l1_loss(color1, img1) + l1_loss(color2, img2)

            # color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)
            loss = reconstruction_loss + 10 * consistency_loss
            val_loss.append(loss.item())
        if random.random() >= 0.8:
            visualize(img1, img2, color1, color2, content1, content2)
        else:
            wandb.log({})
    val_loss = round(sum(val_loss) / len(val_loss), 4)

    checkpoints = {"encoder": encoder.state_dict(),
                   'sNet': sNet.state_dict(),
                   'cNet': cNet.state_dict(),
                   }
    if filter_num == 1:
        checkpoints.update({"filter_color": color_feat1[0:1, :, :, :],
                            "org_color": color_feat2[0:1, :, :, :]})
    torch.save(checkpoints, os.path.join(save_path, f'val_loss_{val_loss}.pth'))


def visualize(I, I_i, Y, Y_i, Z, Z_i):
    idx = 0
    if Y is None and Z_i is None:
        wandb.log({"examples": [
            wandb.Image(to_pil(I[idx].cpu()), caption="I"),
            wandb.Image(to_pil(I_i[idx].cpu()), caption="I_i"),
            wandb.Image(to_pil(Y_i[idx].cpu()), caption="Y_i"),
            wandb.Image(to_pil(Z[idx].cpu()), caption="Z"),

        ]}, commit=False)
    else:
        wandb.log({"examples": [
            wandb.Image(to_pil(I[idx].cpu()), caption="I"),
            wandb.Image(to_pil(I_i[idx].cpu()), caption="I_i"),
            wandb.Image(to_pil(Y[idx].cpu()), caption="Y"),
            wandb.Image(to_pil(Y_i[idx].cpu()), caption="Y_i"),
            wandb.Image(to_pil(Z[idx].cpu()), caption="Z"),
            wandb.Image(to_pil(Z_i[idx].cpu()), caption="Z_i"),
        ]}, commit=False)


def to_pil(tensor):
    unnormalize = transforms.Normalize(
        mean=[-2.12, -2.04, -1.80],
        std=[4.36, 4.46, 4.44]
    )
    tensor = unnormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def image2block(image, patch_size=240, padding=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    patches = []
    # 转换为tensor
    image = transform(image)
    _, H, W = image.shape
    # 上侧、左侧填充padding  右侧和下侧需要计算
    right_padding = padding if W % patch_size == 0 else padding + patch_size - (W % patch_size)
    bottom_padding = padding if H % patch_size == 0 else padding + patch_size - (H % patch_size)
    image = F.pad(image, (padding, right_padding, padding, bottom_padding), mode='replicate')
    row = (image.shape[1] - 2 * padding) // patch_size
    col = (image.shape[2] - 2 * padding) // patch_size
    # 从左到右 从上到下
    for y1 in range(padding, row * patch_size, patch_size):
        for x1 in range(padding, col * patch_size, patch_size):
            patch = image[:, y1 - padding:y1 + patch_size + padding, x1 - padding:x1 + patch_size + padding]
            patch = patch.unsqueeze(0)
            patches.append(patch)
    return patches, row, col


def inference(image, filter,patch_size=240, batch=8, padding=8):
    name = os.path.basename(image)
    image = Image.open(image).convert('RGB')
    width,height = image.size
    split_images, row, col = image2block(image=image, patch_size=patch_size, padding=padding)
    target = Image.new('RGB', (col * patch_size, row * patch_size), 'white')
    content = Image.new('RGB', (col * patch_size, row * patch_size), 'white')
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            batch_input = batch_input.to(device)
            bs,c,w,h = batch_input.shape
            r = filter.repeat(bs, 1, 1, 1)
            d, _ = encoder(batch_input)
            struct = sNet(batch_input, d)
            color = cNet(struct, r)

            struct = struct[:, :, padding:w- padding, padding:h - padding].detach().cpu()
            color = color[:, :, padding:w - padding,padding:h - padding].detach().cpu()

            for j in range(bs):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target.paste(im=to_pil(color[j, :, :, :]), box=(x, y))
                content.paste(im=to_pil(struct[j, :, :, :]), box=(x, y))
    target = target.crop(box=(0, 0, width,height))
    content = content.crop(box=(0, 0, width,height))
    target.save(f'/Users/maoyufeng/Downloads/{name.replace(".jpg", "_nc.jpg")}', quality=100)
    content.save(f'/Users/maoyufeng/Downloads/{name.replace(".jpg", "_content.jpg")}', quality=100)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    # pth = torch.load('pretrained_model/pretrained_model.pth')
    # pth1 = pth['sNet']
    # pth2 = pth['cNet']
    # pth3 = pth['encoder']
    # filter_color = pth["org_color"]
    encoder = Encoder()
    # encoder.load_state_dict(pth3)
    sNet = Shader()
    # sNet.load_state_dict(pth1)
    cNet = Shader()
    # cNet.load_state_dict(pth2)
    encoder.eval()
    sNet.eval()
    cNet.eval()
    encoder.to(device)
    sNet.to(device)
    cNet.to(device)

    img_train, img_val = [], []

    for name in ['classic-neg']:
        # train_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/train'
        # val_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/val'
        train_path = f'/home/dlwork01/slash/{name}/train'
        val_path = f'/home/dlwork01/slash/{name}/val'
        for im in os.listdir(train_path):
            if im.endswith('_org.jpg'):
                img_train.append((os.path.join(train_path, im),
                                  os.path.join(train_path, im.replace("_org", ""))))
        for im in os.listdir(val_path):
            if im.endswith('_org.jpg'):
                img_val.append((os.path.join(val_path, im),
                                os.path.join(val_path, im.replace("_org", ""))))
    train_dataset = ColorTransferDataset(img_train)
    val_dataset = ColorTransferDataset(img_val)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # mscoco_path = '/data/datasets/mscoco/train2017/'
    # mscoco = [os.path.join(mscoco_path, i) for i in os.listdir(mscoco_path) if i.endswith('jpg')]
    # random.shuffle(mscoco)
    # count = len(mscoco)
    # print(f'MSCOCO:{count}')
    #
    # train_dataset = MSCOCODataset(mscoco[:-1000])
    # val_dataset = MSCOCODataset(mscoco[-1000:])
    # train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=24, shuffle=True)

    wandb.init(
        project='filter',
        resume=False,
    )

    train(train_dataloader, val_dataloader, num_epochs=100, learning_rate=3e-4, save_path='pretrained_model')

    # inference(image='/Users/maoyufeng/slash/dataset/org_dataset/classic-neg/DSCF5713_org.jpg',
    #           filter=filter_color)