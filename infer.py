import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
# from torchvision import transforms
from skimage.color import lab2rgb, rgb2lab
from tqdm import tqdm
import argparse
from models import FilterSimulation, FilmMask

# from utils import image_concat

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


def image2block_copy(image, patch_size=640, padding=16):
    # transform = transforms.Compose([
    #     transforms.Resize((patch_size + 2 * padding, patch_size + 2 * padding)),
    #     transforms.ToTensor(),
    # ])
    width, height = image.size
    patches, size_list = [], []
    # 拆分图片
    for left in range(0, width, patch_size):
        for upper in range(0, height, patch_size):
            right = min(left + patch_size, width)
            lower = min(upper + patch_size, height)
            # 裁剪patch_size
            patch = image.crop((left, upper, right, lower))
            # padding解决边缘像素偏移
            patch = cv2.copyMakeBorder(np.array(patch), padding, padding, padding, padding,
                                       borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
            # 记录原始大小，方便后续resize
            size_list.append((patch.shape[1], patch.shape[0]))  # h,w
            # patch = transform(Image.fromarray(patch)).unsqueeze(0)
            patch = torch.from_numpy(cv2.resize(patch, (patch_size, patch_size)) / 255.0).permute(2, 0, 1).unsqueeze(
                0).float()
            # patch = cv2.resize(np.array(patch), (patch_size + 2 * padding, patch_size + 2 * padding))
            # patch = torch.from_numpy(np.clip((rgb2lab(patch / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
            patches.append(patch)
    return patches, size_list


def image2block(image, patch_size=640, padding=16):
    patches, size_list = [], []
    image1 = cv2.copyMakeBorder(np.array(image), padding, padding, padding, padding,
                                borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
    image = Image.fromarray(image1.astype('uint8'))
    width, height = image.size

    for x1 in range(padding, width - 2 * padding, patch_size):
        for y1 in range(padding, height - 2 * padding, patch_size):
            x2 = min(x1 + patch_size + padding, width)
            y2 = min(y1 + patch_size + padding, height)
            patch = np.array(image.crop((x1 - padding, y1 - padding, x2, y2)))
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


def infer(image, model, channels=3, raw=True, patch_size=640, padding=16, batch=8) -> Image:
    """
    image: 输入图片路径
    args: 各种参数
    """
    img = Image.open(image)
    if not raw:
        img = img.resize((patch_size, patch_size))
    # 对每个小块进行推理
    if channels == 3:
        target = Image.new('RGB', img.size)
    else:
        img = img.convert('L')
        target = Image.new('L', img.size)
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
                if len(out.shape)==3:
                    out = out[padding:h - padding, padding:w - padding, :]
                else:
                    out = out[padding:h - padding, padding:w - padding]
                target.paste(Image.fromarray(out), (x, y))
    return target


def init_args():
    parser = argparse.ArgumentParser('Setting For Filter Model', add_help=False)
    parser.add_argument('--model_name', default='Olympus-RichColor', type=str, help='选择的模型名称')
    parser.add_argument('--device', default='cpu', type=str, help='用于图像推理设备')
    parser.add_argument('--batch', default=8, type=int, help='用于设置推理的最大批次')
    parser.add_argument('--patch_size', default=512, type=int, help='和训练输入保持一致')
    parser.add_argument('--padding', default=10, type=int, help='图像边缘padding大小')
    parser.add_argument('--original_output', default=True, type=bool, help='是否采用原始图像大小输出')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pth = torch.load('static/checkpoints/fuji/eterna-bleach-bypass/best.pth', map_location=device)
    channel = pth['decoder.4.bias'].shape[0]
    model = FilterSimulation(channel=channel)
    model.load_state_dict(pth, strict=False)
    model.to(device)
    model.eval()
    st = time.time()
    # Image.open('/Users/maoyufeng/Downloads/DSCF0049_org.jpg').convert('L').save('/Users/maoyufeng/Downloads/DSCF0049_org3.jpg')
    target = infer(image='/Users/maoyufeng/Downloads/DSCF0049_org.jpg', model=model, channels=channel)
    print(time.time() - st)
    # target.show()
    target.save('/Users/maoyufeng/Downloads/DSCF0049_org22.jpg', quality=100)

