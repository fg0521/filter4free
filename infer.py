import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from conf import model_cfg
import argparse


seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #

def image2block(image,args):
    transform = transforms.Compose([
        transforms.Resize((args.patch_size + 2 * args.padding, args.patch_size + 2 * args.padding)),
        transforms.ToTensor(),
    ])
    width, height = image.size
    patches, size_list = [], []
    # 拆分图片
    for left in range(0, width, args.patch_size):
        for upper in range(0, height, args.patch_size):
            right = min(left + args.patch_size, width)
            lower = min(upper + args.patch_size, height)
            # 裁剪patch_size
            patch = image.crop((left, upper, right, lower))
            # padding解决边缘像素偏移
            patch = cv2.copyMakeBorder(np.array(patch), args.padding, args.padding, args.padding, args.padding,borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
            # 记录原始大小，方便后续resize
            size_list.append((patch.shape[1], patch.shape[0]))  # h,w
            patch = transform(Image.fromarray(patch)).unsqueeze(0)
            # patch = cv2.resize(np.array(patch), (patch_size + 2 * padding, patch_size + 2 * padding))
            # patch = torch.from_numpy(np.clip((rgb2lab(patch / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
            patches.append(patch)
    return patches, size_list


def infer(image,args)->Image:
    """
    image: 输入图片路径
    args: 各种参数
    """
    img = Image.open(image)
    if not args.original_output:
        img = img.resize((args.patch_size,args.patch_size))
    # 对每个小块进行推理
    image_size = img.size
    num_cols = image_size[1] //args.patch_size + 1
    target = Image.new('RGB', image_size)
    split_images, size_list = image2block(img, args)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), args.batch),desc=model_cfg[args.model_name]['comment']):
            input = torch.vstack(split_images[i:i + args.batch])
            output = model(input)
            for k in range(output.shape[0]):
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,0).detach().cpu().numpy()
                # channel is lab
                # out = (lab2rgb(output[k, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * [100, 255, 255] - [0, 128,128]) * 255).astype(np.uint8)
                out = cv2.resize(out, size_list[i + k])
                out = out[args.padding:size_list[i + k][1] - args.padding, args.padding:size_list[i + k][0] - args.padding, :]
                row = (i + k) // num_cols
                col = (i + k) % num_cols
                left = col * args.patch_size
                top = row * args.patch_size
                target.paste(Image.fromarray(out), (top, left))
    return target


def init_args():
    parser = argparse.ArgumentParser('Setting For Filter Model', add_help=False)
    parser.add_argument('--model_name',default='Olympus-RichColor',type=str,help='选择的模型名称')
    parser.add_argument('--device',default='cpu',type=str,help='用于图像推理设备')
    parser.add_argument('--batch',default=8,type=int,help='用于设置推理的最大批次')
    parser.add_argument('--patch_size',default=512,type=int,help='和训练输入保持一致')
    parser.add_argument('--padding',default=10,type=int,help='图像边缘padding大小')
    parser.add_argument('--original_output',default=True,type=bool,help='是否采用原始图像大小输出')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    model = model_cfg[args.model_name]['model']
    pth = model_cfg[args.model_name]['checkpoint']
    model.load_state_dict(torch.load(pth, map_location=args.device))
    model.to(args.device)
    target = infer(image='/Users/maoyufeng/slash/dataset/浓郁色调/PA044715.jpg',args=args)
    target.show()
