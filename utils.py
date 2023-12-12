import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


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


def image_concat(img_list, scaled_w=1000, vertical=True):
    w, h = img_list[0].size
    scaled_h = int(scaled_w * h / w)
    if vertical:
        target = Image.new('RGB', (scaled_w, scaled_h * len(img_list)))
    else:
        target = Image.new('RGB', (scaled_w * len(img_list), scaled_h))

    for i, img in enumerate(img_list):
        img = img.resize((scaled_w, scaled_h))
        if vertical:
            target.paste(img, (0, i * scaled_h))
        else:
            target.paste(img, (i * scaled_w, 0))
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
    frame = Image.open('/Users/maoyufeng/Downloads/IMG_0289.PNG')
    img = Image.open(image).resize((3000, 3000))
    W, H = img.size
    ratio = W / H
    # 1:1
    if 0.9 <= ratio < 1.1:
        img = img.resize((3000, 3000))
    # 4:3
    elif 1.1 <= ratio < 1.4:
        img = img.resize((3000, 2250))
    # 3:2
    elif 1.4 <= ratio <= 1.6:
        img = img.resize((3000, 2000))
    # 65:24
    elif 2.6 < ratio < 2.8:
        img = img.resize((3000, 1108))
    frame.paste(img, (200, 200))
    frame.save('/Users/maoyufeng/Downloads/res.png', quality=100)

def drew_loss():
    with open('/Users/maoyufeng/Downloads/用所选项目新建的文件夹/epoch对比/epoch8_train_loss.txt') as f:
        training_loss = [float(i) for i in f.read().split('\n') if i][:155]
    f.close()
    with open('/Users/maoyufeng/Downloads/用所选项目新建的文件夹/epoch对比/epoch16_train_loss.txt') as f:
        eval_loss =[float(i) for i in f.read().split('\n') if i][:155]
    f.close()
    plt.plot(np.array(range(155)), np.array(training_loss), c='r')  # 参数c为color简写，表示颜色,r为red即红色
    plt.plot(np.array(range(len(eval_loss))), np.array(eval_loss), c='b')  # 参数c为color简写，表示颜色,r为red即红色
    plt.legend(labels=['epoch8_train_loss','epoch16_train_loss'])
    # plt.show()
    plt.savefig('/Users/maoyufeng/Downloads/train_loss.png')

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
    drew_loss()