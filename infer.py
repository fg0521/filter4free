import os
import random
import sys
import time
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import FilterSimulation4,  FilterSimulation2, FilterSimulation1, UNet
from utils import color_shift, to_pil

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
# random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #
"""
checkpoint              model
best-v1                 FilterSimulation1
best-v2                 FilterSimulation2
best-v4                 FilterSimulation4
best                    UNet

"""

def image2block(image, patch_size=448, padding=16):
    patches = []
    # 转换为tensor
    image = torch.from_numpy(image / 255.0).float()
    image = image.permute(2, 0, 1)  # c h w
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


def infer(image, model, device, patch_size=448, batch=8, padding=16):
    img = cv2.imread(image) if isinstance(image,str) else image
    channel = 3  # 模型输出的通道数
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if channel==3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
    target = torch.zeros((row * patch_size, col * patch_size, channel), dtype=torch.float)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch],dim=0)
            batch_output = model(batch_input.to(device))
            batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3, 1).cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target[y:y+patch_size, x:x+patch_size] = output
    target = target[:img.shape[0], :img.shape[1]].numpy()
    target = np.clip(target*255,a_min=0,a_max=255).astype(np.uint8)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    return target


def dynamic_infer(image, model, device, patch_size=448, padding=0, batch=8):
    """
    通过滑块来实现动态调整色彩
    """
    img = cv2.imread(image)
    # channel = model.state_dict()['decoder.4.bias'].shape[0] # 获取加载的模型的通道
    channel = 3  # 获取加载的模型的通道
    if channel == 1:
        # 黑白滤镜
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 彩色滤镜
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 对每个小块进行推理
    target = np.zeros(shape=(img.shape), dtype=np.uint8)
    split_images, size_list = image2block(img, patch_size=patch_size, padding=padding)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            input = torch.vstack(split_images[i:i + batch])
            input = input.to(device)
            output = model.forward(input)
            for k in range(output.shape[0]):
                # RGB Channel
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,
                                                                                           0).detach().cpu().numpy()
                x, y, w, h = size_list[i + k]
                out = cv2.resize(out, (w, h))
                out = out[padding:h - padding, padding:w - padding]
                target[y:y + out.shape[0], x:x + out.shape[1]] = out

    return img, target



def compare_model():
    """
    val dataset
    【Model 4V 】	cost:0.026512547181202814s	loss:160.1595167658255
    【Model 4VL】	cost:0.043054302380635186s	loss:160.16032152343755
    """

    model1 = FilterSimulation4()
    model2 = FilterSimulation4L()
    model1.load_state_dict(torch.load('static/checkpoints/fuji/velvia/best-v4.pth',map_location=device))
    model2.load_state_dict(torch.load('static/checkpoints/fuji/velvia/best.pth',map_location=device))
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    path = '/Users/maoyufeng/slash/dataset/train_dataset/velvia/val'
    # path = '/Users/maoyufeng/slash/dataset/org_dataset/velvia'
    loss1,speed1,loss2,speed2 = [],[],[],[]
    for image in os.listdir(path):
        if image.endswith('_org.jpg'):
            org_img = cv2.imread(os.path.join(path,image))
            real_img = cv2.imread(os.path.join(path,image.replace('_org','')))
            if org_img.shape != real_img.shape:
                continue
            t1 = time.time()
            pred_img1 = infer(org_img,model1,device)
            t2 = time.time()
            pred_img2 = infer(org_img,model2,device)
            t3 = time.time()
            speed1.append(t2-t1)
            loss1.append(np.mean(np.abs(pred_img1.astype(np.int16) - real_img.astype(np.int16))))
            speed2.append(t3 - t2)
            loss2.append(np.mean(np.abs(pred_img2.astype(np.int16) - real_img.astype(np.int16))))

    print(f"【Model 4V 】\tcost:{sum(speed1)/len(speed1)}s\tloss:{sum(loss1)/len(loss1)}")
    print(f"【Model 4VL】\tcost:{sum(speed2)/len(speed2)}s\tloss:{sum(loss2)/len(loss2)}")


def image2block4unet(image, patch_size=240, padding=8):
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


def infer4unet(image, model, device, patch_size=448, batch=8, padding=16):
    image = Image.open(image).convert('RGB')
    w, h = image.size
    split_images, row, col = image2block4unet(image=image, patch_size=patch_size, padding=padding)
    target = Image.new('RGB', (col * patch_size, row * patch_size), 'white')
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            batch_output = model(batch_input.to(device))
            batch_output = batch_output[:, :, padding:batch_output.shape[2] - padding,
                           padding:batch_output.shape[3] - padding].detach().cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target.paste(im=to_pil(output), box=(x, y))
    target = target.crop(box=(0, 0, w, h))
    return target


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pth1 = torch.load('static/checkpoints/fuji/provia/best.pth', map_location=device)
    model = UNet()
    model.load_state_dict(pth1)
    model.to(device)
    model.eval()
    st = time.time()

    target = infer4unet(image=f'/Users/maoyufeng/slash/dataset/org_dataset/provia100f/DSCF1492_org.JPG',
                        model=model,
                        device=device,
                        padding=8,
                        )
    print(time.time() - st)
    # cv2.imwrite(f'/Users/maoyufeng/Downloads/14234.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, 100])
    target.save('/Users/maoyufeng/Downloads/14234.jpg',quality=100)
    # app = QApplication(sys.argv)
    # window = Demo('/Users/maoyufeng/Downloads/iShot_2024-02-06_16.35.23.png')
    # window.show()
    # sys.exit(app.exec_())

    # compare_model()
