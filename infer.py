import os
import random
import sys
import time
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
# from torchvision import transforms
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QImage, QPixmap
from tqdm import tqdm
from models import FilterSimulation,  FilterSimulationFast, DNCM, Encoder
# from PyQt5.QtWidgets import QSlider, QPushButton, QLabel, QApplication, QMainWindow, QWidget, QVBoxLayout

from utils import color_shift

seed = 2333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
# random.seed(seed)  # random and transforms
torch.backends.cudnn.deterministic = True  #


def image2block_slow(image, patch_size=448, padding=16):
    patches, size_list = [], []
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                               borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101

    H, W, C = image.shape
    # width, height = image.size
    # 从上到下 从左到右
    for x1 in range(padding, W - 2 * padding, patch_size):
        for y1 in range(padding, H - 2 * padding, patch_size):
            x2 = min(x1 + patch_size + padding, W)
            y2 = min(y1 + patch_size + padding, H)
            # patch = np.array(image.crop((x1 - padding, y1 - padding, x2, y2)))
            patch = image[y1 - padding:y2, x1 - padding:x2, :]
            size_list.append((x1 - padding, y1 - padding, patch.shape[1], patch.shape[0]))  # x,y,w,h
            # RGB Channel
            patch = torch.from_numpy(cv2.resize(patch, (patch_size, patch_size)) / 255.0)
            if len(patch.shape) == 2:
                patch = patch.unsqueeze(-1)
            patch = patch.permute(2, 0, 1).unsqueeze(0).float()
            # # LAB Channel
            # patch = cv2.resize(patch,(patch_size,patch_size))
            # # RGB->LAB and softmax to [0,1]
            # patch = np.clip((rgb2lab(patch / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            # patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
            patches.append(patch)
    return patches, size_list


def infer_slow(image, model, device, patch_size=448, batch=8, padding=16):
    """
    infer+image2block
    使用opencv进行填充、切片、resize，记录下标和宽高进行拼接和还原
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

                # # LAB Channel
                # out = (lab2rgb(output[k, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * [100, 255, 255] - [0, 128,128]) * 255).astype(np.uint8)
                x, y, w, h = size_list[i + k]
                out = cv2.resize(out, (w, h))
                out = out[padding:h - padding, padding:w - padding]

                target[y:y + out.shape[0], x:x + out.shape[1]] = out
    return cv2.cvtColor(target, cv2.COLOR_RGB2BGR)


def image2block(image, patch_size=448, padding=16):
    patches = []
    # 转换为tensor
    image = torch.from_numpy(image / 255.0)
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
            patch = patch.unsqueeze(0).float()
            patches.append(patch)
    return patches, row, col


def infer(image, model, device, patch_size=448, batch=8, padding=16):
    img = cv2.imread(image)
    channel = 3  # 模型输出的通道数
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if channel==3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
    target = torch.zeros((row * patch_size, col * patch_size, channel), dtype=torch.uint8)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch)):
            batch_input = torch.cat(split_images[i:i + batch],dim=0)
            batch_output = model(batch_input.to(device))
            batch_output = torch.clamp(batch_output * 255, min=0, max=255).byte()
            batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3, 1).cpu()
            for j, output in enumerate(batch_output):
                y = (i + j) // col * patch_size
                x = (i + j) % col * patch_size
                target[y:y+patch_size, x:x+patch_size] = output
    target = target[:img.shape[0], :img.shape[1]].numpy()
    return cv2.cvtColor(target, cv2.COLOR_RGB2BGR)


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


# class Demo(QMainWindow):
#     def __init__(self,image_path):
#         super().__init__()
#         self.setGeometry(100,100,800, 600)
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setStyleSheet(
#             "QSlider::groove:horizontal {"
#             "border: 1px solid gray;"
#             "height: 5px;"
#             "left: 10px;"
#             "right: 20px;}"
#             "QSlider::handle:horizontal {"
#             "border: 1px solid gray;"
#             "background:white;"
#             "border-radius: 7px;"
#             "width: 14px;"
#             "height:14px;"
#             "margin: -6px;}"
#             "QSlider::add-page:horizontal{background: #3a3c42;}"
#             "QSlider::sub-page:horizontal{background: #b2596f; }"
#         )
#         self.img_input = QLabel('Drag Image Here')
#         window_layout = QVBoxLayout()
#         window_layout.addWidget(self.img_input)
#         window_layout.addWidget(self.slider)
#         central_widget = QWidget()
#         central_widget.setLayout(window_layout)
#         self.setCentralWidget(central_widget)
#         self.display(image_path)
#         img,target = dynamic_infer(image=image_path,
#                                model=model1,
#                                device=device,
#                                )
#         self.slider.setValue(100)
#         self.slider.setMaximum(200)
#         self.slider.setMinimum(0)
#         self.slider.setSingleStep(10)
#
#         self.slider.valueChanged.connect(lambda :self.predict(img,target))
#
#     def predict(self,img,target):
#         temp = self.slider.value()/100
#         print(f'Temperature:{temp}')
#         target = torch.tensor((1.0 - temp) * (img/255.0) + temp * (target/255.0))
#         target = torch.clamp(target*255, min=0, max=255).numpy().astype(np.uint8)
#         # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
#         self.display(target)
#
#     def display(self,image):
#         if isinstance(image,str):
#             image = cv2.imread(image)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image,(self.img_input.width(),self.img_input.height()))
#         q_image = QImage(image[:], image.shape[1],image.shape[0], image.shape[1] * 3,
#                              QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(q_image)
#         self.img_input.setPixmap(pixmap)
#         self.img_input.setAlignment(Qt.AlignCenter)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pth1 = torch.load('/Users/maoyufeng/Downloads/classic-neg2/best.pth', map_location=device)
    model = FilterSimulation()
    model.load_state_dict(pth1)
    model.to(device)
    model.eval()
    st = time.time()
    target = infer(image='/Users/maoyufeng/slash/dataset/富士/nc/test/DSCF0281_org.jpg',
                    model=model,
                    device=device,
                    padding=8,
                    )
    print(time.time() - st)
    cv2.imwrite('/Users/maoyufeng/Downloads/input7.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # app = QApplication(sys.argv)
    # window = Demo('/Users/maoyufeng/Downloads/iShot_2024-02-06_16.35.23.png')
    # window.show()
    # sys.exit(app.exec_())
