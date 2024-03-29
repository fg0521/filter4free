import math
import random

import cv2
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def images2gif(dir, gif_name, resize=(200, 200)):
    # 打开图像文件并将它们添加到图像列表中
    images = []
    for img_name in sorted(os.listdir(dir)):
        if img_name.endswith('jpg'):
            image = Image.open(os.path.join(dir, img_name))
            image = image.resize(resize)
            images.append(image)

    # 将图像列表保存为 GIF 动画
    images[0].save(
        os.path.join(dir, f"{gif_name}.gif"),
        save_all=True,
        append_images=images[1:],
        duration=100,  # 每帧的持续时间（以毫秒为单位）
        loop=0  # 设置为0表示无限循环
    )


def image2hsit(img, show=False):
    # 读取RGB图像
    image = cv2.imread(img)  # 替换为你的图像文件路径
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 计算直方图
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    # 将直方图归一化到0到1之间
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()
    # 将直方图堆叠成一个张量
    hist_tensor = np.hstack((hist_r, hist_g, hist_b))
    if show:
        # 可视化直方图
        plt.ylim(0, 0.15)
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.show()
    return hist_tensor


def image_concat(img_list, scaled_w=None, vertical=True):
    w, h = img_list[0].size
    if scaled_w is not None:
        w = scaled_w
        h = int(scaled_w * h / w)

    if vertical:
        target = Image.new('RGB', (w, h * len(img_list)))
    else:
        target = Image.new('RGB', (w * len(img_list), h))

    for i, img in enumerate(img_list):
        img = img.resize((w, h))
        if vertical:
            target.paste(img, (0, i * h))
        else:
            target.paste(img, (i * w, 0))
    return target


def add_chosen_status(org_img):
    name = os.path.basename(org_img).split('.')[0]
    chosen_img = Image.open('static/src/chosen.jpg').resize((25, 25))
    org_img = Image.open(org_img).resize((100, 100))
    org_img.save(f'static/src/{name}-ORG.png')
    org_img.paste(im=chosen_img, box=(75, 75))
    org_img.convert(mode='RGBA')
    org_img.save(f'static/src/{name}.png')


def add_frame(image):
    H, W, C = image.shape
    maxWH = max(W, H)
    minWH = int(math.log10(min(W, H)) * 0.12 * min(W, H))
    padding = int(maxWH * 0.02)
    background = np.zeros((H + padding + minWH, W + 2 * padding, 3), dtype=np.uint8)
    background.fill(255)
    background[padding:padding + H, padding:padding + W] = image
    cv2.imshow('test', background)
    cv2.waitKey(0)


def drew_loss(filter_name, train, eval):
    with open(train) as f:
        train_loss = [float(i) for i in f.read().split('\n') if i]
    f.close()
    with open(eval) as f:
        eval_loss = [float(i) for i in f.read().split('\n') if i]
    f.close()
    plt.plot(np.array(range(len(train_loss))), np.array(train_loss), c='r')  # 参数c为color简写，表示颜色,r为red即红色
    plt.plot(np.array(range(len(eval_loss))), np.array(eval_loss), c='b')  # 参数c为color简写，表示颜色,r为red即红色
    plt.legend(labels=['train_loss', 'eval_loss'])
    plt.xlabel(filter_name)
    path = os.path.dirname(train)
    plt.savefig(os.path.join(path, f'{filter_name}_loss.png'))


def cal_l1loss(model, org_path, real_path, checkpoint_path):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    target = infer(image=org_path, model=model, channels=3, device=device)
    target_tensor = torch.tensor(np.asarray(target), dtype=torch.float32)
    real_tensor = torch.tensor(cv2.cvtColor(cv2.imread(real_path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
    l1_loss = F.l1_loss(real_tensor, target_tensor)
    return l1_loss


def awb(image):
    img = cv2.imread(image)
    b, g, r = cv2.split(img)  # 图像bgr通道分离
    avg_b = np.average(b)
    avg_g = np.average(g)
    avg_r = np.average(r)
    k = (avg_b + avg_g + avg_r) / 3  # 计算k值
    kr = k / avg_r  # 计算rgb的增益(增益通常在0-2的浮点数之间)
    kb = k / avg_b
    kg = k / avg_g
    # 根据增益逐个调整RGB三通道的像素值，超出部分取255（数据类型还是要转换回uint8）
    new_b = np.where((kb * b) > 255, 255, kb * b).astype(np.uint8)
    new_g = np.where((kg * g) > 255, 255, kg * g).astype(np.uint8)
    new_r = np.where((kr * r) > 255, 255, kr * r).astype(np.uint8)
    # 合并三个通道
    img_new = cv2.merge([new_b, new_g, new_r])
    cv2.imshow('img', img)
    cv2.imshow('img_new', img_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_shift(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    offset = random.randint(-10,10)
    image[:, :, 0] = (image[:, :, 0] + offset) % 180
    image[:, :, 0] = np.clip(image[:, :, 0], 0, 179)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    blur_seed = 2*min(int(random.random()*3),1)+1
    image = cv2.GaussianBlur(image , (blur_seed, blur_seed), 0)
    print(offset)
    print(blur_seed)
    return image


if __name__ == '__main__':
    # images2gif(dir='test/canon',
    #            gif_name='canon')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/org.jpg')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/small-rgb.jpg')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/small-lab.jpg',show=True)

    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/test2.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/19416.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/small-rgb-new2.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/small-rgb-new3.jpg',show=True)
    # drew_loss(filter_name='nostalgic-neg',
    #           train='static/checkpoints/fuji/nostalgic-neg/train_loss.txt',
    #           eval='static/checkpoints/fuji/nostalgic-neg/eval_loss.txt')
    #
    # loss = cal_l1loss(org_path='/Users/maoyufeng/slash/dataset/org_dataset/common/DSCF0132_org.jpg',
    #                   real_path='/Users/maoyufeng/slash/dataset/org_dataset/superia400/DSCF0132.jpg',
    #                   checkpoint_path='static/checkpoints/fuji/superia400/best.pth')
    # print(loss)
    # im1 = Image.open('/Users/maoyufeng/slash/dataset/富士/nc/test/DSCF0268_org.jpg')
    # im2 = Image.open('/Users/maoyufeng/slash/dataset/富士/nc/test/DSCF0268_NC.jpg')
    # im3 = Image.open('/Users/maoyufeng/slash/dataset/富士/nc/test/DSCF0268.JPG')
    # res = image_concat(img_list=[im1,im2,im3])
    # res.save('/Users/maoyufeng/slash/dataset/富士/nc/test/res4.jpg',quality=100)
    image = cv2.imread('/Users/maoyufeng/Downloads/iShot_2024-02-05_16.12.24.png')
    add_frame(image=image)
