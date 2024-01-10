import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
# from torchvision import transforms
from tqdm import tqdm
from models import FilterSimulation
seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #



def image2block(image, patch_size=448, padding=16):
    patches, size_list = [], []
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
    H,W,C = image.shape
    # image = Image.fromarray(image1.astype('uint8'))
    # width, height = image.size
    # 从上到下 从左到右
    for x1 in range(padding, W - 2 * padding, patch_size):
        for y1 in range(padding, H - 2 * padding, patch_size):
            x2 = min(x1 + patch_size + padding, W)
            y2 = min(y1 + patch_size + padding, H)
            # patch = np.array(image.crop((x1 - padding, y1 - padding, x2, y2)))
            patch= image[y1 - padding:y2,x1 - padding:x2,:]
            size_list.append((x1 - padding, y1 - padding, patch.shape[1], patch.shape[0]))  # x,y,w,h
            # RGB Channel
            patch = torch.from_numpy(cv2.resize(patch, (patch_size, patch_size)) / 255.0)
            if len(patch.shape) == 2:
                patch = patch.unsqueeze(-1)
            patch = patch.permute(2, 0, 1).unsqueeze(0).float()
            # # LAB Channel
            # patch = cv2.resize(patch,(patch_size,patch_size))
            # # RGB->LAB and softmax to [0,1]
            # patch = np.clip((rgb2lab(patch / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            # patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
            patches.append(patch)

    return patches, size_list


def infer(image, model, device, patch_size=448, padding=16, batch=8):
    """
    image: 输入图片路径
    args: 各种参数
    """
    img = cv2.imread(image)
    channel = model.state_dict()['decoder.4.bias'].shape[0] # 获取加载的模型的通道
    if channel ==1:
        # 黑白滤镜
        if img.shape[2]==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # 彩色滤镜
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 对每个小块进行推理
    target = np.zeros(shape=(img.shape),dtype=np.uint8)
    split_images, size_list = image2block(img, patch_size=patch_size, padding=padding)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            input = torch.vstack(split_images[i:i + batch])
            input = input.to(device)
            output = model(input)
            for k in range(output.shape[0]):
                # RGB Channel
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,0).detach().cpu().numpy()
                # # LAB Channel
                # out = (lab2rgb(output[k, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * [100, 255, 255] - [0, 128,128]) * 255).astype(np.uint8)
                x, y, w, h = size_list[i + k]
                out = cv2.resize(out, (w, h))
                out = out[padding:h - padding, padding:w - padding]
                target[y:y+out.shape[0],x:x+out.shape[1]]= out
    return cv2.cvtColor(target,cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pth = torch.load('static/checkpoints/fuji/superia400/best.pth', map_location=device)
    model = FilterSimulation()
    model.load_state_dict(pth)
    model.to(device)
    model.eval()
    st = time.time()
    target = infer(image='/Users/maoyufeng/Downloads/WechatIMG27.jpg', model=model,device=device)
    print(time.time() - st)
    cv2.imwrite('/Users/maoyufeng/Downloads/23213332.jpg',target,[cv2.IMWRITE_JPEG_QUALITY, 100])
