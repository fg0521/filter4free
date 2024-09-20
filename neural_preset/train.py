import itertools

import random
from model import Encoder,DNCM
import torch
import torch.optim as optim
from PIL import Image
from pillow_lut import load_cube_file
from torch.optim.lr_scheduler import LambdaLR
import wandb
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #



class MyDataset(Dataset):
    def __init__(self, img_list,lut_path='/data/datasets/LUTS/'):
        self.img_list = img_list
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.luts_path = lut_path
        self.luts = [os.path.join(self.luts_path,i) for i in os.listdir(self.luts_path) if i.endswith('.cube')]

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


def lr_lambda(epoch):
    return 1 if epoch < 20 else 0.1


def train(train_dataloader,val_dataloader, num_epochs, learning_rate, device,lambda_num=10,
          val_epoch=3,save_path='./checkpoint'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    encoder.to(device)
    sDNCM.to(device)
    nDNCM.to(device)
    optimizer = optim.Adam(params=itertools.chain(encoder.parameters(),sDNCM.parameters(),nDNCM.parameters()) , lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    step = 0
    for epoch in range(num_epochs):
        encoder.train()
        sDNCM.train()
        nDNCM.train()
        for I_i, I_j in tqdm(train_dataloader,desc=f"Epoch:{epoch+1}"):
            I_i, I_j = I_i.to(device), I_j.to(device)
            optimizer.zero_grad()

            d_i,r_i = encoder(I_i) # get content1_feature/style1_feature
            d_j, r_j = encoder(I_j)    # get content2_feature/style2_feature

            Z_i = nDNCM(I=I_i,T=d_i)   # img1 + content1_feature = img1_content
            Z_j = nDNCM(I=I_j,T=d_j)   # img2 + content2_feature = img2_content

            # 注意这里是交叉输入
            Y_i = sDNCM(I=Z_j,T=r_i)
            Y_j = sDNCM(I=Z_i,T=r_j)

            consistency_loss = l2_loss(Z_i, Z_j)
            reconstruction_loss = l1_loss(I_i,Y_i) + l1_loss(I_j,Y_j)

            msg = {
                "reconstruction_loss": reconstruction_loss.item(),
                "consistency_loss":consistency_loss.item(),
            }
            step+=1
            if step%100==0:
                visualize(I_i, I_j, Z_i,Z_j,Y_i,Y_j)
            else:
                wandb.log({})
            loss = reconstruction_loss + lambda_num * consistency_loss
            wandb.log(msg, commit=False)
            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save({
            'encoder': encoder.state_dict(),
            'sDNCM': sDNCM.state_dict(),
            'nDNCM': nDNCM.state_dict()
        }, os.path.join(save_path, f'model_train_{epoch}.pth'))
        if (epoch + 1) % val_epoch == 0:
            val(val_dataloader=val_dataloader,
                device=device,
                save_path=save_path,lambda_num=lambda_num)




def val(val_dataloader,device,save_path,lambda_num):
    encoder.eval()
    sDNCM.eval()
    nDNCM.eval()
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    val_loss = []

    for I_i, I_j in tqdm(val_dataloader):
        I_i, I_j = I_i.to(device), I_j.to(device)
        with torch.no_grad():

            d_i, r_i = encoder(I_i)  # get content1_feature/style1_feature
            d_j, r_j = encoder(I_j)  # get content2_feature/style2_feature

            Z_i = nDNCM(I=I_i, T=d_i)  # img1 + content1_feature = img1_content
            Z_j = nDNCM(I=I_j, T=d_j)  # img2 + content2_feature = img2_content

            Y_i = sDNCM(I=Z_j, T=r_i)
            Y_j = sDNCM(I=Z_i, T=r_j)

            consistency_loss = l2_loss(Z_i, Z_j)
            reconstruction_loss = l1_loss(I_i, Y_i) + l1_loss(I_j, Y_j)

            loss = reconstruction_loss + lambda_num * consistency_loss
            val_loss.append(loss.item())

            if random.random() >= 0.8:
                visualize(I_i, I_j, Z_i,Z_j,Y_i,Y_j)
            else:
                wandb.log({})
        l = round(sum(val_loss) / len(val_loss), 4)
        torch.save({
            'encoder':encoder.state_dict(),
            'sDNCM':sDNCM.state_dict(),
            'nDNCM':nDNCM.state_dict()
        }, os.path.join(save_path, f'model_eval_loss_{l}.pth'))


def visualize(I_i, I_j, Z_i,Z_j,Y_i,Y_j):
    idx = 0
    wandb.log({"examples": [
        wandb.Image(to_pil(I_i[idx].cpu()), caption="image1"),
        wandb.Image(to_pil(I_j[idx].cpu()), caption="image2"),
        wandb.Image(to_pil(Z_i[idx].cpu()), caption="content1"),
        wandb.Image(to_pil(Z_j[idx].cpu()), caption="content2"),
        wandb.Image(to_pil(Y_i[idx].cpu()), caption="style1"),
        wandb.Image(to_pil(Y_j[idx].cpu()), caption="style2"),
    ]}, commit=False)




def to_pil(tensor):
    unnormalize = transforms.Normalize(
        mean=[-2.12, -2.04, -1.80],
        std=[4.36, 4.46, 4.44]
    )
    tensor = unnormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)




if __name__ == '__main__':
    encoder = Encoder()
    sDNCM = DNCM()
    nDNCM = DNCM()
    img_train,img_val = [],[]


    # for name in ['classic-neg']:
    #     # train_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/train'
    #     # val_path = '/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/val'
    #     train_path = f'/home/dlwork01/slash/{name}/train'
    #     val_path = f'/home/dlwork01/slash/{name}/val'
    #     for im in os.listdir(train_path):
    #         if im.endswith('_org.jpg'):
    #             img_train.append((os.path.join(train_path, im),
    #                               os.path.join(train_path, im.replace("_org", ""))))
    #     for im in os.listdir(val_path):
    #         if im.endswith('_org.jpg'):
    #             img_val.append((os.path.join(val_path, im),
    #                             os.path.join(val_path, im.replace("_org", ""))))
    mscoco_path ='/data/datasets/mscoco/train2017/'
    mscoco = [os.path.join(mscoco_path,i) for i in os.listdir(mscoco_path) if i.endswith('jpg')]
    random.shuffle(mscoco)
    count = len(mscoco)
    print(f'MSCOCO:{count}')

    train_dataset = MyDataset(mscoco[:-1000])
    val_dataset = MyDataset(mscoco[-1000:])
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=24, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    pth = torch.load('./pretrain/model_train_2.pth',map_location=device)
    encoder.load_state_dict(pth['encoder'])
    sDNCM.load_state_dict(pth['sDNCM'])
    nDNCM.load_state_dict(pth['nDNCM'])

    wandb.init(
        project='neural_preset',
        resume=False,
    )
    train(train_dataloader, val_dataloader, num_epochs=30, learning_rate=3e-4,
                 device=device, save_path='./pretrain')


