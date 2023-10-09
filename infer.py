import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
from models import FilmMask2, Olympus, Unet
ttf = ImageFont.truetype('/Users/maoyufeng/shuzheng/project/algo-workshop/generator/identityCard/src/hei.ttf', 80)


def image2block(image,patch_size,padding):
    transform = transforms.Compose([
        transforms.Resize((patch_size+2*padding,patch_size+2*padding)),
        transforms.ToTensor(),
    ])
    width, height = image.size
    patches,size_list = [],[]
    # 拆分图片
    for left in range(0, width, patch_size):
        for upper in range(0, height, patch_size):
            right = min(left + patch_size, width)
            lower = min(upper + patch_size, height)
            # 裁剪patch_size
            patch = image.crop((left, upper, right, lower))
            # padding解决边缘像素偏移
            patch = cv2.copyMakeBorder(np.array(patch), padding, padding, padding, padding, borderType=cv2.BORDER_REPLICATE) # cv2.BORDER_REFLECT_101
            # 记录原始大小，方便后续resize
            size_list.append((patch.shape[1],patch.shape[0]))   # h,w
            patch = Image.fromarray(patch)
            patches.append(transform(patch).unsqueeze(0))
    return patches,size_list


def block2image(results, image_size, patch_size,size_list,padding):
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
        out = torch.clamp(result * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy()
        # resize 到原始block大小
        out = cv2.resize(out,size_list[i])
        # 裁剪得到padding之前的图片
        out = out[padding:size_list[i][1]-padding,padding:size_list[i][0]-padding,:]
        im = Image.fromarray(out)
        # ImageDraw.Draw(org).rectangle([(top,left), (top+80, left+80)], fill=None, outline="red", width=2)
        # ImageDraw.Draw(org).text((top,left), str(i), font=ttf, fill=(255, 0, 0))
        # 拼接图片
        target.paste(im, (top, left))
    return target


def infer(image, patch_size=512,padding=10,batch_size=1):
    img = Image.open(image)
    results = []
    # 对每个小块进行推理
    split_images,size_list = image2block(img, patch_size,padding)
    for i in tqdm(range(0,len(split_images),batch_size)):
        input = torch.vstack(split_images[i:i+batch_size])
        output = model(input)
        [results.append(output[k,:,:,:]) for k in range(output.shape[0])]
    target = block2image(results, image_size=img.size, patch_size=patch_size,size_list=size_list,padding=padding)
    return target


def image_concat(true_img, train_img, scaled_w=1000):
    w, h = true_img.size
    scaled_h = int(scaled_w * h / w)
    target = Image.new('RGB', (scaled_w, scaled_h * 2))
    true_img = train_img.resize((scaled_w, scaled_h))
    train_img = train_img.resize((scaled_w, scaled_h))
    target.paste(true_img, (0, 0))
    target.paste(train_img, (0, scaled_h))
    target.show()





if __name__ == '__main__':
    model = FilmMask2()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('checkpoints/film_mask/Copy of best.pth', map_location=device))
    model.to(device)
    target = infer(image='/Users/maoyufeng/Downloads/iShot_2023-10-09_14.31.24.jpg',
                   batch_size=8)
    target.show()
