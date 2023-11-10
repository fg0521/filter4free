import cv2
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt



def images2gif(dir,gif_name):
    # 打开图像文件并将它们添加到图像列表中
    images = []
    for img_name in sorted(os.listdir(dir)):
        if img_name.endswith('jpg'):
            image = Image.open(os.path.join(dir,img_name))
            images.append(image)

    # 将图像列表保存为 GIF 动画
    images[0].save(
        os.path.join(dir,f"{gif_name}.gif"),
        save_all=True,
        append_images=images[1:],
        duration=100,  # 每帧的持续时间（以毫秒为单位）
        loop=0 # 设置为0表示无限循环
    )


def image2hsit(img,show=False):
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
        plt.ylim(0,0.15)
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.show()
    return hist_tensor

def image_concat(img_list, scaled_w=1000,vertical=True):
    w, h = img_list[0].size
    scaled_h = int(scaled_w * h / w)
    if vertical:
        target = Image.new('RGB', (scaled_w, scaled_h * len(img_list)))
    else:
        target = Image.new('RGB', (scaled_w * len(img_list),scaled_h))

    for i,img in enumerate(img_list):
        img = img.resize((scaled_w, scaled_h))
        if vertical:
            target.paste(img, (0, i*scaled_h))
        else:
            target.paste(img, (i*scaled_w,0))
    return target


if __name__ == '__main__':
    images2gif(dir='test/fuji/velvia',
               gif_name='velvia')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/org.jpg')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/small-rgb.jpg')
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/small-lab.jpg',show=True)

    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/test2.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/19416.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/small-rgb-new2.jpg',show=True)
    # image2hsit(img='/Users/maoyufeng/slash/dataset/色罩/test/small-rgb-new3.jpg',show=True)

