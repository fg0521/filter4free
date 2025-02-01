import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import rawpy
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


def image2block(image, patch_size=448, padding=16):
    patches = []
    # 转换为tensor
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std  # 归一化
    image = torch.from_numpy(image).float()
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
    img_input = cv2.imread(image) if isinstance(image, str) else image
    channel = 3  # 模型输出的通道数
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) if channel == 3 else cv2.cvtColor(img_input,
                                                                                             cv2.COLOR_BGR2GRAY)
    split_images, row, col = image2block(img_input, patch_size=patch_size, padding=padding)
    img_output = torch.zeros((row * patch_size, col * patch_size, channel), dtype=torch.float)
    with torch.no_grad():
        for i in range(0, len(split_images), batch):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            batch_output = model(batch_input.to(device))
            batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3, 1).cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                img_output[y:y + patch_size, x:x + patch_size] = output
    img_output = img_output[:img_input.shape[0], :img_input.shape[1]].numpy()
    mean = np.array([-2.12, -2.04, -1.80])
    std = np.array([4.36, 4.46, 4.44])
    img_output = (img_output - mean) / std
    img_output = np.clip(img_output * 255, a_min=0, a_max=255).astype(np.uint8)
    # img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
    return img_input, img_output


def read_image(file_path)->np.ndarray:
    file_type = Path(file_path).suffix.lower()
    if file_type in ['.raf', '.cr2', '.cr3', '.arw', '.rw2', '.dng', '.pef']:
        with rawpy.imread(file_path) as raw:
            image = raw.postprocess(use_camera_wb=True)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        image = Image.open(file_path).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

if __name__ == '__main__':
    for img in os.listdir('/Users/maoyufeng/Downloads/test'):
        im = read_image(f'/Users/maoyufeng/Downloads/test/{img}')
        print(img+'  okk')