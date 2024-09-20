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
from models import UNet, FilterSimulation4, UCM, Shader,Encoder
from tqdm import tqdm

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
transform  = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
transform2  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def normalize(array):
    return (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

def unnormalize(array):
    return (array - np.array([-2.12, -2.04, -1.80])) / np.array([4.36, 4.46, 4.44])


def image2block(image, patch_size=448, padding=16, norm=True):
    patches = []
    image = image / 255.0
    image = normalize(image) if norm else image
    # 获取图像的高和宽
    H, W, C = image.shape
    # 计算右侧和下侧的填充量
    right_padding = padding if W % patch_size == 0 else padding + patch_size - (W % patch_size)
    bottom_padding = padding if H % patch_size == 0 else padding + patch_size - (H % patch_size)
    # 填充图像
    padded_image = np.pad(image, ((padding, bottom_padding), (padding, right_padding), (0, 0)), mode='edge')
    row = (padded_image.shape[0] - 2 * padding) // patch_size
    col = (padded_image.shape[1] - 2 * padding) // patch_size
    # 生成patches
    for y1 in range(padding, row * patch_size + padding, patch_size):
        for x1 in range(padding, col * patch_size + padding, patch_size):
            patch = padded_image[y1 - padding:y1 + patch_size + padding, x1 - padding:x1 + patch_size + padding]
            patch = np.expand_dims(patch.transpose(2, 0, 1), 0).astype(np.float32)
            patches.append(patch)
    return patches, row, col


def infer(image, model, device, patch_size=448, batch=8, padding=16, norm=True):
    img = cv2.imread(image) if isinstance(image, str) else image
    channel = 3  # 模型输出的通道数
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if channel == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding, norm=norm)
    target = torch.zeros((row * patch_size, col * patch_size, channel), dtype=torch.float)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            # batch_input = torch.cat(split_images[i:i + batch], dim=0).to(device)
            batch_input = torch.from_numpy(np.vstack(split_images[i:i + batch])).to(device)
            batch_output = model(batch_input)
            batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3, 1).cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target[y:y + patch_size, x:x + patch_size] = output
    target = target[:img.shape[0], :img.shape[1]].numpy()
    if norm:
        target = unnormalize(target)
    target = np.clip(target * 255, a_min=0, a_max=255).astype(np.uint8)
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
    from test import UNet, UNetStudent
    from models import FilterSimulation4
    model1 = FilterSimulation4()
    model2 = UNetStudent()
    model1.load_state_dict(torch.load('static/checkpoints/kodak/gold200/best-v4.pth', map_location=device))
    model2.load_state_dict(torch.load('/Users/maoyufeng/Downloads/unet/best-v4.pth', map_location=device))
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    path = '/Users/maoyufeng/slash/dataset/train_dataset/gold200/val'
    # path = '/Users/maoyufeng/slash/dataset/org_dataset/velvia'
    loss1, speed1, loss2, speed2 = [], [], [], []
    for i, image in enumerate(os.listdir(path)):
        if image.endswith('_org.jpg'):
            org_img = cv2.imread(os.path.join(path, image))
            real_img = cv2.imread(os.path.join(path, image.replace('_org', '')))
            if org_img.shape != real_img.shape:
                continue
            t1 = time.time()
            pred_img1 = infer(org_img, model1, device)
            t2 = time.time()
            pred_img2 = infer(org_img, model2, device)
            t3 = time.time()
            cv2.imwrite(f'/Users/maoyufeng/Downloads/test/{i}_FilterSimulation.jpg', pred_img1,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(f'/Users/maoyufeng/Downloads/test/{i}_UNet.jpg', pred_img2, [cv2.IMWRITE_JPEG_QUALITY, 100])
            speed1.append(t2 - t1)
            loss1.append(np.mean(np.abs(pred_img1.astype(np.int16) - real_img.astype(np.int16))))
            speed2.append(t3 - t2)
            loss2.append(np.mean(np.abs(pred_img2.astype(np.int16) - real_img.astype(np.int16))))

    print(f"【Model 4V 】\tcost:{sum(speed1) / len(speed1)}s\tloss:{sum(loss1) / len(loss1)}")
    print(f"【Model UNet】\tcost:{sum(speed2) / len(speed2)}s\tloss:{sum(loss2) / len(loss2)}")



def image2block2(image, patch_size=448, padding=16):
    patches = []
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

def infer2(tgt,checkpoint,src=None,patch_size=240, batch=8, padding=8, norm=True):
    """
    :param image:
    :param checkpoint:
    :param filter_image:
    :param patch_size:
    :param batch:
    :param padding:
    :param norm:
    :return:
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pth = torch.load(checkpoint,map_location=device)
    encoder = Encoder()
    sNet = Shader()
    cNet = Shader()
    encoder.load_state_dict(pth['encoder'])
    sNet.load_state_dict(pth['sNet'])
    cNet.load_state_dict(pth['cNet'])
    encoder.eval()
    sNet.eval()
    cNet.eval()
    encoder.to(device)
    sNet.to(device)
    cNet.to(device)

    tgt_img_cp = Image.open(tgt).convert('RGB')
    tgt_img_cp = transform2(tgt_img_cp)
    tgt_img = tgt_img_cp.unsqueeze(0)
    tgt_img = tgt_img.to(device)

    # 使用预设的lut进行推理
    if src is None:
        filter_color = pth['filter_color']
        # with torch.no_grad():
        #     d, _ = encoder(tgt_img)

    # 使用选中的图像进行提取lut再推理
    elif src is not None:
        src_img = Image.open(src).convert('RGB')
        src_img = transform2(src_img)
        src_img = src_img.unsqueeze(0)
        src_img = src_img.to(device)
        with torch.no_grad():
            _, filter_color = encoder(src_img)
            d, _ = encoder(tgt_img)
    else:
        print('Have No LUT For Inference...')
    split_images, row, col = image2block2(tgt_img_cp, patch_size=patch_size, padding=padding)
    target = Image.new(mode='RGB',size=(row * patch_size, col * patch_size))
    content = Image.new(mode='RGB',size=(row * patch_size, col * patch_size))
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch], dim=0)
            batch_input = batch_input.to(device)
            bs = batch_input.shape[0]
            r = filter_color.repeat(bs, 1, 1, 1)
            # _, r = encoder(batch_input)
            # d = d.repeat(bs,1,1,1)
            d, _ = encoder(batch_input)
            struct = sNet(batch_input, d)
            color = cNet(struct, r)
            for j in range(bs):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target.paste(im=to_pil(color[j, :, :, :]),box=(x,y))
                content.paste(im=to_pil(struct[j, :, :, :]),box=(x,y))
                # content[y:y + patch_size, x:x + patch_size] = to_pil(struct[j:j + 1, :, :, :])
    # target = target[:tgt_img_cp.shape[0], :tgt_img_cp.shape[1]].numpy()
    # content = content[:tgt_img_cp.shape[0], :tgt_img_cp.shape[1]].numpy()
    # if norm:
    #     target = normalize(target)
        # content = normalize(content)
    # target = np.clip(target * 255, a_min=0, a_max=255).astype(np.uint8)
    # content = np.clip(content * 255, a_min=0, a_max=255).astype(np.uint8)
    # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    # content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    target.save('output.jpg', quality=100)
    content.save('content.jpg', quality=100)





if __name__ == '__main__':
    model_list = {
        'UNet': {'model': UNet(), 'pth': 'best.pth'},
        'FilterSimulation': {'model': FilterSimulation4(), 'pth': 'best-v4.pth'}
    }

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model_name = 'FilterSimulation'
    model = model_list[model_name]['model']
    # model = FilterSimulation2iPhone()
    pth = torch.load(f'pack/static/checkpoints/fuji/classic-neg/best-v4.pth', map_location=device)
    model.load_state_dict(pth)
    model.to(device)
    model.eval()
    st = time.time()
    target = infer(image=f'/Users/maoyufeng/Downloads/iShot_2024-08-28_17.39.31.jpg',
                   model=model,
                   device=device,
                   padding=8,
                   patch_size=640,
                   # norm=False
                   )
    print(time.time() - st)
    cv2.imwrite(f'/Users/maoyufeng/Downloads/output-pytorch.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # compare_model()

    # infer2(tgt='/Users/maoyufeng/Downloads/iShot_2024-08-28_17.39.31.jpg',
    #           checkpoint='test_checkpoint/val_loss_0.35.pth')