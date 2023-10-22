import math
import os
import random
import time
import cv2
import numpy as np
from tqdm import tqdm


def laplacian(img, size=1):
    kernel = np.array([[-1, -1, -1], [-1, size + 8, -1], [-1, -1, -1]])
    res = cv2.filter2D(img, -1, kernel)
    res[res > 255] = 255
    res[res < 0] = 0
    return res.astype(np.uint8)


class Processor():

    def __init__(self, mode='random'):
        self.mode = mode  # random：随机裁剪  order：顺序裁剪
        self.clip_size = 512
        self.clip_num = 30
        self.max_clip_size = 1200

    def add_rotate(self, image: list):
        """
        对list中的image做相同变换
        包括旋转90、180、270、水平、垂直反转
        """
        op = random.randint(0, 4)
        if op == 0:
            rotate_image = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in image]
        elif op == 1:
            rotate_image = [cv2.rotate(img, cv2.ROTATE_180) for img in image]
        elif op == 2:
            rotate_image = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in image]
        elif op == 3:
            rotate_image = [cv2.flip(img, 0) for img in image]
        else:
            rotate_image = [cv2.flip(img, 1) for img in image]
        return rotate_image

    def add_sharpen(self, image: list):
        """
        清晰度变化：锐化、模糊
        """
        op = random.randint(0, 3)
        if op == 0:
            sharpen_list = [cv2.GaussianBlur(img, (7, 7), 0) for img in image]
        elif op == 1:
            sharpen_list = [cv2.blur(img, (5, 5)) for img in image]
        elif op == 2:
            sharpen_list = [cv2.medianBlur(img, 5) for img in image]
        else:
            sharpen_list = [laplacian(img) for img in image]
        return sharpen_list

    def add_concat(self,image:list):
        """
        图像拼接
        [[原图拼接图像],[目标图片拼接图像]]
        """
        if int(math.sqrt(len(image[0])))**2 == len(image[0]):
            h,w,c = image[0][0].shape
            n = int(math.sqrt(len(image[0])))
            # 创建一个空白图像，用于 n x n 拼接
            result = np.zeros((w * n, h * n, 3), dtype=np.uint8)
            for each_image in image:
                images = []
                for i in range(0,len(each_image),n):
                    images.append(each_image[i,i+n])
                # 拼接图像
                for i in range(n):
                    for j in range(n):
                        result[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[i][j]


    def run(self, input_path, output_path, splitting=True, rotate=True, sharpen=True,concat=True):
        file_list = os.listdir(input_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mode in ['train', 'val']:
            if not os.path.exists(os.path.join(output_path, mode)):
                os.mkdir(os.path.join(output_path, mode))
        # 对超大分辨率图片进行裁剪
        if splitting:
            for org_img_name in tqdm(file_list,desc='数据增强'):
                if org_img_name.lower().endswith('_org.jpg'):
                    img_name = org_img_name.replace('_org', '')
                    org_im = cv2.imread(os.path.join(input_path, org_img_name))
                    goal_im = cv2.imread(os.path.join(input_path, img_name))
                    assert org_im.shape == goal_im.shape, 'shapes are not equal!'
                    H, W, C = org_im.shape
                    if self.mode == 'random':
                        # 每张图片随机裁剪 clip-num 次
                        for i in range(self.clip_num):
                            goal_name = str(time.time()).replace('.', '')
                            org_name = goal_name + '_org.jpg'
                            goal_name = goal_name + '.jpg'
                            mode = 'val' if i > int(self.clip_num * 0.9) else 'train'
                            x, y = random.randint(0, W - self.clip_size), random.randint(0, H - self.clip_size)
                            w, h = random.randint(self.clip_size, self.max_clip_size), random.randint(self.clip_size,
                                                                                                 self.max_clip_size)
                            ww = W if x + w > W else x + w
                            hh = H if y + h > H else y + h

                            org_im_split = org_im[y:hh, x:ww]
                            goal_im_split = goal_im[y:hh, x:ww]

                            if random.random() >= 0.5 and rotate:
                                org_im_split, goal_im_split = self.add_rotate([org_im_split, goal_im_split])
                            if random.random() >= 0.5 and sharpen:
                                org_im_split, goal_im_split = self.add_sharpen([org_im_split, goal_im_split])

                            cv2.imwrite(os.path.join(output_path, mode, org_name), org_im_split)
                            cv2.imwrite(os.path.join(output_path, mode, goal_name), goal_im_split)
                    elif self.mode == 'order':
                        cnt, x = 0, 1
                        while x <= W:
                            x += self.clip_size
                            y = 0
                            while y <= H:
                                goal_name = str(time.time()).replace('.', '')
                                org_name = goal_name + '_org.jpg'
                                goal_name = goal_name + '.jpg'
                                y += self.clip_size
                                file = 'val' if cnt % 10 == 0 else 'train'
                                ww = W if x > W else x
                                hh = H if y > H else y
                                org_im_split = org_im[y - self.clip_size:hh, x - self.clip_size:ww]
                                goal_im_split = goal_im[y - self.clip_size:hh, x - self.clip_size:ww]

                                if random.random() >= 0.5 and rotate:
                                    org_im_split, goal_im_split = self.add_rotate([org_im_split, goal_im_split])
                                if random.random() >= 0.5 and sharpen:
                                    org_im_split, goal_im_split = self.add_sharpen([org_im_split, goal_im_split])

                                cv2.imwrite(os.path.join(output_path, file, org_name), org_im_split)
                                cv2.imwrite(os.path.join(output_path, file, goal_name), goal_im_split)
                                cnt += 1



if __name__ == '__main__':

    p = Processor()
    p.run(input_path='/Users/maoyufeng/Downloads/1',
          output_path='/Users/maoyufeng/Downloads/test')

    # im1 = cv2.imread('/Users/maoyufeng/slash/dataset/色罩/org/9.jpg')
    # im2 = cv2.imread('/Users/maoyufeng/slash/dataset/色罩/org/9_mask.jpg')
    # H,W,_ = im1.shape
    # w = int(W/5)
    # for i in range(0,W,w):
    #     cv2.imwrite(f'/Users/maoyufeng/slash/dataset/色罩/9{i}.jpg',im1[:H,i:i+w,:])
    #     cv2.imwrite(f'/Users/maoyufeng/slash/dataset/色罩/9{i}_org.jpg',im2[:H,i:i+w,:])
