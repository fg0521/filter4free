
"""
Neural Preset for Color Style Transfer
https://arxiv.org/pdf/2303.13511
个人复现
"""

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
from models import  UCM
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


class ColorTransferDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.luts_path = '/data/datasets/LUTS/'
        # self.luts = [i for i in os.listdir(self.luts_path) if i.endswith('.cube')]
        self.luts = []

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if len(self.img_list[idx])==2:
            # [(org,filter)]
            img1,img2 = self.img_list[idx]
            img1 = Image.open(img1).convert(mode='RGB')
            img2 = Image.open(img2).convert(mode='RGB')
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1_path = self.img_list[idx]
            img1 = Image.open(img1_path).convert(mode='RGB')
            lut = os.path.join(self.luts_path,random.choice(self.luts))
            lut = load_cube_file(lut)
            img2 = img1.filter(lut)

            img1 = self.transform(img1)

            if random.random() < 0.5:
                enhancer = ImageEnhance.Color(img2)
                factor = random.uniform(0.5, 1.5)  # 随机饱和度因子
                img2 = enhancer.enhance(factor)
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(img2)
                factor = random.uniform(0.5, 1.5)  # 随机对比度因子
                img2 = enhancer.enhance(factor)
            img2 = self.transform(img2)

        return img1, img2


def lr_lambda(epoch):
    return 1 if epoch < 24 else 0.1


def train(train_dataloader,val_dataloader, num_epochs, learning_rate, device,
          val_epoch=3,save_path='./checkpoint'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.to(device)
    optimizer = optim.Adam(params=list(model.parameters()) , lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda)

    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    rgb_loss = RGBLoss()
    for epoch in range(num_epochs):
        model.train()
        for img1, img2 in tqdm(train_dataloader,desc=f"Epoch:{epoch+1}"):
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            org_content, f_content, s1, s2, Y, Y_i = model(img1,img2)

            consistency_loss = l2_loss(org_content, f_content)
            reconstruction_loss = l1_loss(Y_i, img2) + l1_loss(Y, img1)
            # TODO add RGBLoss to help fit loss
            color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)

            msg = {
                "reconstruction_loss": reconstruction_loss.item(),
                'rgb_loss': color_loss.item(),
                "consistency_loss":consistency_loss.item(),
            }

            loss = reconstruction_loss + 10 * consistency_loss + color_loss
            wandb.log(msg, commit=False)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (epoch + 1) % val_epoch == 0:
            val(val_dataloader=val_dataloader,
               model=model,
                device=device,
                save_path=save_path)


def train_on_one(train_dataloader,val_dataloader, num_epochs, learning_rate, device,
          val_epoch=3,save_path='./checkpoint'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.to(device)
    # params = list(model.cNet.parameters())+list(model.S.parameters())+list(model.S_upsample.parameters())
    params = list(model.cNet.parameters())+list(model.sNet.parameters())
    optimizer = optim.Adam(params=params , lr=learning_rate, weight_decay=1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda)

    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    rgb_loss = RGBLoss()
    step = 0

    for epoch in range(num_epochs):
        model.cNet.train()
        # model.S_upsample.train()
        # model.S.train()
        model.sNet.train()
        for img1, img2 in tqdm(train_dataloader,desc=f"Epoch:{epoch+1}"):
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            org_content,f_content,s1,s2,Y,Y_i = model(org_img=img1,filter_img=img2)

            consistency_loss = l2_loss(org_content, f_content)
            reconstruction_loss = l1_loss(Y_i, img2) + l1_loss(Y, img1)
            # reconstruction_loss = l1_loss(Y_i, img2)
            # TODO add RGBLoss to help fit loss
            color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)
            # color_loss = rgb_loss(Y_i, img2)
            loss = reconstruction_loss + color_loss+consistency_loss
            msg = {
                "reconstruction_loss": reconstruction_loss.item(),
                'rgb_loss': color_loss.item(),
                "consistency_loss":consistency_loss.item()
            }
            wandb.log(msg, commit=False)
            step+=1
            if step % 100 == 0:
                visualize(img1, img2, Y, Y_i, org_content, f_content)
            else:
                wandb.log({})
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % val_epoch == 0:
            val_on_one(val_dataloader=val_dataloader,
                model=model,
                device=device,
                save_path=save_path)



def val(val_dataloader,model,device,save_path):
    model.eval()
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    rgb_loss = RGBLoss()
    val_loss = []
    for img1, img2 in tqdm(val_dataloader):
        img1, img2 = img1.to(device), img2.to(device)
        with torch.no_grad():
            org_content, f_content, s1, s2, Y, Y_i = model(img1,img2)  # 原始图像的结构信息+色彩信息
            consistency_loss = l2_loss(org_content, f_content)
            reconstruction_loss = l1_loss(Y_i, img2) + l1_loss(Y, img1)
            color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)

            loss = reconstruction_loss + 10 * consistency_loss + color_loss
            val_loss.append(loss.item())
        if random.random()>=0.8:
            visualize(img1, img2, Y, Y_i, org_content, f_content)
        else:
            wandb.log({})
    val_loss = round(sum(val_loss) / len(val_loss),4)
    torch.save(model.state_dict(), os.path.join(save_path, f'UCM_loss_{val_loss}.pth'))


def val_on_one(val_dataloader, model, device, save_path):
    model.cNet.eval()
    # model.S_upsample.eval()
    # model.S.eval()
    model.sNet.eval()
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    rgb_loss = RGBLoss()
    val_loss = []
    for img1, img2 in tqdm(val_dataloader):
        img1, img2 = img1.to(device), img2.to(device)
        with torch.no_grad():
            org_content,f_content,s1,s2,Y,Y_i = model(org_img=img1,filter_img=img2)
            consistency_loss = l2_loss(org_content, f_content)
            reconstruction_loss = l1_loss(Y_i, img2) + l1_loss(Y, img1)
            # reconstruction_loss = l1_loss(Y_i, img2)
            # TODO add RGBLoss to help fit loss
            color_loss = rgb_loss(Y_i, img2) + rgb_loss(Y, img1)
            # color_loss = rgb_loss(Y_i, img2)
            loss = reconstruction_loss + color_loss +consistency_loss

            val_loss.append(loss.item())
        if random.random()>=0.8:
            visualize(img1, img2, Y, Y_i, org_content, f_content)
        else:
            wandb.log({})
    val_loss = round(sum(val_loss) / len(val_loss), 4)

    torch.save({
        'filter':s2[:1,:,:,:],
        'cNet':model.cNet.state_dict(),
        'sNet':model.sNet.state_dict(),
    },
    os.path.join(save_path,f'filter_loss_{val_loss}.pth'))


def visualize(I, I_i, Y,Y_i,Z,Z_i):
    idx = 0
    if Y is None and  Z_i is None:
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
    wandb.log({})


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



def inference(model,filter,image,patch_size=240, batch=8,padding=8):
    name = os.path.basename(image)
    image = Image.open(image).convert('RGB')
    w,h = image.size
    split_images, row, col = image2block(image=image,patch_size=patch_size,padding=padding)
    target = Image.new('RGB', (col * patch_size,row * patch_size), 'white')
    content = Image.new('RGB', (col * patch_size,row * patch_size), 'white')
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            struct, _, _, _, batch_output, _ = model(org_img=batch_input.to(device), filter=filter)
            batch_output = batch_output[:, :, padding:batch_output.shape[2]-padding, padding:batch_output.shape[3]-padding].detach().cpu()
            struct = struct[:, :, padding:struct.shape[2]-padding, padding:struct.shape[3]-padding].detach().cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target.paste(im=to_pil(output),box=(x,y))
                content.paste(im=to_pil(struct[j,:,:,:]),box=(x,y))
    target = target.crop(box=(0,0,w,h))
    content = content.crop(box=(0,0,w,h))
    target.save(f'/Users/maoyufeng/Downloads/{name.replace(".jpg","_nc.jpg")}',quality=100)
    content.save(f'/Users/maoyufeng/Downloads/{name.replace(".jpg","_content.jpg")}',quality=100)









if __name__ == '__main__':

    model = UCM(continue_train=True)
    img_train,img_val = [],[]
    train_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/train'
    val_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/val'
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

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    wandb.init(
        project='test',
        resume=False,
    )
    model.load_state_dict(torch.load('pretrained_model.pth',map_location=device))
    # train(train_dataloader, val_dataloader,num_epochs=32, learning_rate=3e-4, device=device,
    #       save_path='./test_checkpoint')

    train_on_one(train_dataloader, val_dataloader, num_epochs=200, learning_rate=1e-4,
                 device=device, save_path='./test_checkpoint2')



    pth = torch.load('test_checkpoint2/filter_loss_0.1902.pth',map_location=device)
    filter_vec = pth['filter']
    model.sNet.load_state_dict(pth['sNet'])
    model.cNet.load_state_dict(pth['cNet'])
    model.to(device)
    model.eval()
    st = time.time()
    inference(model,filter_vec,'')
    print(time.time()-st)
