import copy
import math
import os
import random
import shutil
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

    def __init__(self, mode='order', clip_size=512,align_size=30):
        self.mode = mode  # random：随机裁剪  order：顺序裁剪
        self.clip_size = clip_size+2*align_size
        self.align_size = align_size
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

    def add_concat(self, image: list):
        """
        图像拼接
        [原图拼接图像]/[目标拼接图像]
        """
        if len(image) >= 9:
            image = image[:9]
            n = 3
        elif len(image) >= 4:
            image = image[:4]
            n = 2
        else:
            image = image[:1]
            n = 1
        h, w, c = image[0].shape
        result = np.zeros((w * n, h * n, 3), dtype=np.uint8)
        # 拼接图像
        for i in range(n):
            for j in range(n):
                result[i * h:(i + 1) * h, j * w:(j + 1) * w] = image[i * n + j]
        return result

    def add_resize(self, org_im, goal_im):
        if org_im.shape > goal_im.shape:
            org_im = cv2.resize(org_im, goal_im.shape[:2][::-1])
        elif org_im.shape < goal_im.shape:
            goal_im = cv2.resize(goal_im, org_im.shape[:2][::-1])
        return org_im, goal_im

    def run(self, input_path, output_path, clip=True, concat=False, rotate=True, sharpen=False,
            align=False, min_byte=50.0):
        file_list = sorted([i for i in os.listdir(input_path) if i.lower().endswith('_org.jpg')])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for mode in ['train', 'val']:
            if not os.path.exists(os.path.join(output_path, mode)):
                os.mkdir(os.path.join(output_path, mode))
        if clip:
            cnt = 0
            for org_img_name in tqdm(file_list, desc='数据裁剪'):
                img_name = org_img_name.replace('_org', '')
                org_im = cv2.imread(os.path.join(input_path, org_img_name))
                goal_im = cv2.imread(os.path.join(input_path, img_name))
                if org_im is None or goal_im is None:
                    continue
                if org_im.shape != goal_im.shape:
                    org_im, goal_im = self.add_resize(org_im=org_im, goal_im=goal_im)
                H, W, C = org_im.shape

                train, val = [], []
                if self.mode == 'random':
                    # 每张图片随机裁剪 clip-num 次
                    for i in range(self.clip_num):
                        x, y = random.randint(0, W - self.clip_size), random.randint(0, H - self.clip_size)
                        w, h = random.randint(self.clip_size, self.max_clip_size), random.randint(self.clip_size,
                                                                                                  self.max_clip_size)
                        ww = W if x + w > W else x + w
                        hh = H if y + h > H else y + h
                        org_im_split = org_im[y:hh, x:ww]
                        goal_im_split = goal_im[y:hh, x:ww]
                        if i <= int(self.clip_num * 0.9):
                            train.append((org_im_split, goal_im_split))
                        else:
                            val.append((org_im_split, goal_im_split))
                elif self.mode == 'order':
                    for x in range(0, W, self.clip_size)[:-1]:
                        for y in range(0, H, self.clip_size)[:-1]:
                            org_im_split = org_im[y:y+self.clip_size, x:x + self.clip_size]
                            goal_im_split = goal_im[y:y+self.clip_size, x:x + self.clip_size]
                            if org_im_split.any() and goal_im_split.any():
                                cnt += 1
                                if cnt % 10 == 0:
                                    val.append((org_im_split, goal_im_split))
                                else:
                                    train.append((org_im_split, goal_im_split))

                for mode in ['train', 'val']:
                    for org_im, goal_im in eval(mode):
                        if align:
                            org_im_cp = org_im[self.align_size:org_im.shape[0] - self.align_size, self.align_size:org_im.shape[1] - self.align_size, :]
                            org_im = self.add_align(org_img=org_im, goal_img=goal_im)
                            org_im = org_im[self.align_size:org_im.shape[0] - self.align_size, self.align_size:org_im.shape[1] - self.align_size, :]
                            goal_im = goal_im[self.align_size:goal_im.shape[0] - self.align_size, self.align_size:goal_im.shape[1] - self.align_size, :]
                            diff = abs(np.sum(org_im_cp/255.0-org_im/255.0))
                            if diff==0 or diff>3e3 or np.sum(org_im==0)>3000:
                                continue


                        if random.random() >= 0.5 and rotate:
                            org_im, goal_im = self.add_rotate([org_im, goal_im])
                        if random.random() >= 0.5 and sharpen:
                            org_im, goal_im = self.add_sharpen([org_im, goal_im])

                        name = str(time.time()).replace('.', '')
                        cv2.imwrite(os.path.join(output_path, mode, name + '_org.jpg'), org_im)
                        cv2.imwrite(os.path.join(output_path, mode, name + '.jpg'), goal_im)
                        # if os.path.getsize(os.path.join(output_path, mode, name + '_org.jpg')) / 1024 < min_byte:
                        #     os.remove(os.path.join(output_path, mode, name + '_org.jpg'))
                        #     os.remove(os.path.join(output_path, mode, name + '.jpg'))
        if concat:
            for i in tqdm(range(500), desc='数据拼接'):
                img_num = set()
                while len(img_num) < 9:
                    img_num.add(random.randint(0, len(file_list) - 1))
                org_list, goal_list = [], []
                for num in img_num:
                    org_im = cv2.imread(os.path.join(input_path, file_list[num]))
                    goal_im = cv2.imread(os.path.join(input_path, file_list[num].replace('_org', '')))
                    x, y = random.randint(0, org_im.shape[1] - 1000), random.randint(0, org_im.shape[0] - 1000)
                    org_im = org_im[y:y + 250, x:x + 250]
                    goal_im = goal_im[y:y + 250, x:x + 250]
                    if random.random() >= 0.5 and rotate:
                        org_im, goal_im = self.add_rotate([org_im, goal_im])
                    if random.random() >= 0.5 and sharpen:
                        org_im, goal_im = self.add_sharpen([org_im, goal_im])
                    if org_im.shape == (250, 250, 3) and goal_im.shape == (250, 250, 3):
                        org_list.append(org_im)
                        goal_list.append(goal_im)
                if len(org_list) == 9:
                    org_im = self.add_concat(org_list)
                    goal_im = self.add_concat(goal_list)
                    name = str(time.time()).replace('.', '')
                    mode = 'val' if i % 10 == 0 else 'train'
                    cv2.imwrite(os.path.join(output_path, mode, name + '_org.jpg'), org_im)
                    cv2.imwrite(os.path.join(output_path, mode, name + '.jpg'), goal_im)

    def add_align(self, org_img, goal_img):
        # 读取两张图像
        if isinstance(org_img, str):
            image_org = cv2.imread(org_img)
            image_goal = cv2.imread(goal_img)
        elif isinstance(org_img, np.ndarray):
            image_org = org_img
            image_goal = goal_img
        # 初始化SIFT检测器
        sift = cv2.SIFT_create()

        # 找到关键点和特征描述
        keypoints1, descriptors1 = sift.detectAndCompute(image_org, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image_goal, None)

        # 使用FLANN匹配器来匹配特征描述
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        if descriptors1 is None or descriptors2 is None:
            return image_org
        elif descriptors1.shape[0]<2 or descriptors2.shape[0]<2:
            return image_org
        else:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)

            # 选择好的匹配
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # 提取关键点的坐标
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            if len(src_pts) > 4 and len(dst_pts) > 4:
                # 使用RANSAC算法进行变换估计
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is None:
                    return image_org
                else:
                    # 使用得到的变换矩阵进行透视变换
                    image_org_aligned = cv2.warpPerspective(image_org, M, (image_goal.shape[1], image_goal.shape[0]))
                    return image_org_aligned
            else:
                return image_org


if __name__ == '__main__':
    p = Processor(mode='order', clip_size=800,align_size=0)
    p.run(input_path=f'/Users/maoyufeng/Downloads/Set2_ground_truth_images',
          output_path=f'/Users/maoyufeng/slash/dataset/train_dataset/awb',
          min_byte=0.0, concat=False, clip=True, align=False, sharpen=False)

    # '1704176476307436'