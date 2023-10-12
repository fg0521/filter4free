import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from skimage.color import lab2rgb, rgb2lab
from torchvision import transforms
from tqdm import tqdm
from models import FilmMaskBase, Olympus, Unet, FilmMaskSmall,FilmMaskTiny
from distillation import StudentModel

ttf = ImageFont.truetype('/Users/maoyufeng/shuzheng/project/algo-workshop/generator/identityCard/src/hei.ttf', 80)
from conf import model_cfg


def image2block(image, patch_size, padding, channel):
    transform = transforms.Compose([
        transforms.Resize((patch_size + 2 * padding, patch_size + 2 * padding)),
        transforms.ToTensor(),
    ])
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
            if channel == 'rgb':
                patch = transform(Image.fromarray(patch)).unsqueeze(0)
            else:
                patch = cv2.resize(np.array(patch), (patch_size + 2 * padding, patch_size + 2 * padding))
                patch = torch.from_numpy(
                    np.clip((rgb2lab(patch / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)).permute(2, 0, 1).to(
                    torch.float32).unsqueeze(0)
            patches.append(patch)

    return patches, size_list


def block2image(results, image_size, patch_size, size_list, padding):
    num_cols = image_size[1] // patch_size + 1
    target = Image.new('RGB', image_size)
    # 遍历每个小块的结果并将其拼接到完整图像中
    for i, result in enumerate(results):
        row = i // num_cols
        col = i % num_cols
        left = col * patch_size
        top = row * patch_size
        # right = left + patch_size
        # bottom = top + patch_size
        # print(top,left,right,bottom)
        # out = torch.clamp(result * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy()
        # resize 到原始block大小
        # out = cv2.resize(out,size_list[i])
        # 裁剪得到padding之前的图片
        # out = out[padding:size_list[i][1]-padding,padding:size_list[i][0]-padding,:]
        im = Image.fromarray(result)
        # ImageDraw.Draw(org).rectangle([(top,left), (top+80, left+80)], fill=None, outline="red", width=2)
        # ImageDraw.Draw(org).text((top,left), str(i), font=ttf, fill=(255, 0, 0))
        # 拼接图片
        target.paste(im, (top, left))
    return target


def infer(image, patch_size=512, padding=10, batch_size=1, channel='rgb'):
    # global out
    img = Image.open(image)
    # results = []
    # 对每个小块进行推理
    image_size = img.size
    num_cols = image_size[1] // patch_size + 1
    target = Image.new('RGB', image_size)
    split_images, size_list = image2block(img, patch_size, padding, channel)
    for i in tqdm(range(0, len(split_images), batch_size)):
        input = torch.vstack(split_images[i:i + batch_size])
        output = model(input)
        for k in range(output.shape[0]):
            if channel == 'rgb':
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,
                                                                                           0).detach().cpu().numpy()
            else:
                out = (lab2rgb(output[k, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * [100, 255, 255] - [0, 128,
                                                                                                               128]) * 255).astype(
                    np.uint8)
            out = cv2.resize(out, size_list[i + k])
            out = out[padding:size_list[i + k][1] - padding, padding:size_list[i + k][0] - padding, :]

            row = (i + k) // num_cols
            col = (i + k) % num_cols
            left = col * patch_size
            top = row * patch_size
            target.paste(Image.fromarray(out), (top, left))
            # results.append(out)
    # target = block2image(results, image_size=img.size, patch_size=patch_size,size_list=size_list,padding=padding)
    return target


if __name__ == '__main__':
    model_name = 'film_mask'
    model = model_cfg[model_name]['model']
    pth = model_cfg[model_name]['checkpoint']
    # model = FilmMaskBase()
    # pth = 'checkpoints/film_mask/epoch1.pth'

    device = torch.device('cpu')
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device)
    target = infer(image='/Users/maoyufeng/slash/dataset/色罩/test/test2.jpg',
                   batch_size=8,
                   channel='rgb')
    # # org = Image.open('/Users/maoyuf eng/slash/dataset/浓郁色调/PA044706_mask.jpg')
    # # image_concat(org,target)
    target.save('/Users/maoyufeng/slash/dataset/色罩/test/small-rgb-new2.jpg')
    # '/Users/maoyufeng/Downloads/3.jpg'
    # im0 = Image.open('/Users/maoyufeng/Downloads/7.jpg')
    # im1 = Image.open('/Users/maoyufeng/Downloads/4.jpg')
    # im2 = Image.open('/Users/maoyufeng/Downloads/5.jpg')
    # im3 = Image.open('/Users/maoyufeng/Downloads/6.jpg')
    # image_concat([im0,im1,im2,im3],vertical=False).save('/Users/maoyufeng/Downloads/color.jpg')
