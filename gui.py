import os
import sys
import threading

import cv2
import torch
from PIL import Image, ImageEnhance
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QButtonGroup, QRadioButton, \
    QScrollArea, QPushButton, QListWidget, QListWidgetItem, QLabel, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPixmap, QIcon
from functools import partial

from infer import infer, init_args, image2block
from models import FilterSimulation


class PredictionWorker(QObject):
    update_progress = pyqtSignal(int)

    def predict(self, image):
        self.model = FilterSimulation()
        self.device = torch.device('cpu')
        print(f'开始预测：{image}')
        self.model.load_state_dict(
            torch.load('checkpoints/olympus/rich-color/best.pth', map_location=self.device))
        self.model.to(self.device)
        img = Image.open(image)
        # 对每个小块进行推理
        image_size = img.size
        num_cols = image_size[1] // 512 + 1
        target = Image.new('RGB', image_size)
        split_images, size_list = image2block(img, patch_size=512, padding=10)

        t = min(100 / len(split_images) * 8,100)
        print(t)
        cnt = 1
        with torch.no_grad():
            for i in range(0, len(split_images), 8):
                input = torch.vstack(split_images[i:i + 8])
                output = self.model(input)
                for k in range(output.shape[0]):
                    out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,
                                                                                               0).detach().cpu().numpy()
                    out = cv2.resize(out, size_list[i + k])
                    out = out[10:size_list[i + k][1] - 10,
                          10:size_list[i + k][0] - 10, :]
                    row = (i + k) // num_cols
                    col = (i + k) % num_cols
                    left = col * 512
                    top = row * 512
                    target.paste(Image.fromarray(out), (top, left))
                self.update_progress.emit(int(t * cnt))
                cnt += 1

        save_path = os.path.dirname(image)
        name = '.'.join(os.path.basename(image).split('.')[:-1]) + '_predict.jpg'
        target.save(os.path.join(save_path, name))
        # print(f'保存到：{os.path.join(save_path, name)}')


class PredictionThread(QThread):
    def __init__(self, image):
        super().__init__()
        self.worker = PredictionWorker()
        self.image = image

    def run(self):
        self.worker.predict(self.image)


class FilmGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.set_ui()

    def set_ui(self):
        self.setWindowTitle("滤镜模拟器")
        self.setGeometry(100, 100, 600, 480)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 预测图片
        self.predict_image = ''

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(40, 120, 40, 40)

        # 图片拖动显示框
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # 启用 QLabel 接受拖放事件
        self.image_label.setAcceptDrops(True)
        self.image_label.setStyleSheet(
            "border: 1px solid black;"
            "background-color: white;"  # 背景颜色
        )

        # 左侧滚动区域用于选择滤镜
        filter_chosen = QScrollArea()
        filter_chosen.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        filter_chosen.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        filter_chosen.setFixedWidth(108)
        self.image_input = QWidget()

        # 设置垂直分布
        filter_layout = QVBoxLayout()

        self.filter_button_group = QButtonGroup()
        # 存放按钮和对应的图片路径
        self.filter_buttons = []
        filter_list = [i for i in os.listdir('src') if i.endswith("ORG.PNG")]
        for filter_name in filter_list:  # 创建20个按钮，模拟滚动效果
            button = QPushButton()
            button.setFixedHeight(80)  # 设置按钮高度
            button.setFixedWidth(80)
            button.setStyleSheet(
                "QPushButton {"
                f"   border-image: url({os.path.join('src', filter_name)});"  # 背景颜色
                "}"

            )
            self.filter_buttons.append((button, filter_name))
            # 按钮的滤镜选择事件
            button.clicked.connect(lambda state, button=button: self.choose4filters(button))
            self.filter_button_group.addButton(button)
            filter_layout.addWidget(button)

        self.image_input.setLayout(filter_layout)
        filter_chosen.setWidget(self.image_input)

        main_layout.addWidget(filter_chosen)
        # main_layout.addWidget(self.image_label)

        # 右侧文件上传区域
        upload_layout = QVBoxLayout()
        upload_layout.addWidget(self.image_label)

        # 右下角的两个按钮
        button_bar_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.predict_button = QPushButton("Start")
        self.predict_button.clicked.connect(self.start_prediction)

        self.predict_button.setFixedHeight(50)

        button_bar_layout.addWidget(self.progress_bar)
        button_bar_layout.addWidget(self.predict_button)
        upload_layout.addLayout(button_bar_layout)
        main_layout.addLayout(upload_layout)
        self.central_widget.setLayout(main_layout)
        self.setAcceptDrops(True)

    # 滤镜选择
    def choose4filters(self, clicked_button):
        for button, filter_name in self.filter_buttons:
            if button is clicked_button:
                button.setStyleSheet("QPushButton {"
                                     f"   border-image: url({os.path.join('src', filter_name.replace('_ORG', ''))});"  # 背景颜色
                                     "}")
            else:
                button.setStyleSheet("QPushButton {"
                                     f"   border-image: url({os.path.join('src', filter_name)});"  # 背景颜色
                                     "}")

    # 拖拽事件
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.predict_image = url.toLocalFile()
                self.display4image(self.predict_image)

    # 显示图片
    def display4image(self,image):
        print(f'显示：{image}')
        pixmap = QPixmap(image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), transformMode=Qt.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignCenter)

    # 开始预测函数
    def start_prediction(self):
        self.prediction_thread = PredictionThread(self.predict_image)
        self.prediction_thread.worker.update_progress.connect(self.update_progress_bar)
        self.predict_button.setEnabled(False)
        self.prediction_thread.start()
    
    # 更新进度条
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.predict_button.setEnabled(True)
            save_path = os.path.dirname(self.predict_image)
            name = '.'.join(os.path.basename(self.predict_image).split('.')[:-1]) + '_predict.jpg'
            print(f'结束：{os.path.join(save_path, name)}')
            self.display4image(os.path.join(save_path, name))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilmGUI()
    window.show()
    sys.exit(app.exec_())

    # for img in os.listdir('/Users/maoyufeng/slash/fuji-chrome'):
    #     if img.endswith('PNG'):
    #         im = Image.open(f'/Users/maoyufeng/slash/fuji-chrome/{img}')
    #         im = im.resize((100,100))
    #         im.save(f'./src/{img}')
    #         colorEnhancer = ImageEnhance.Color(im)
    #         im = colorEnhancer.enhance(0.5)
    #         im.save(f'./src/{img.replace(".PNG","_ORG.PNG")}')
