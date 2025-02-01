import os
import sys
import time
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QMainWindow, QButtonGroup, \
    QScrollArea, QPushButton, QLabel, QMessageBox, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QComboBox, \
    QFileDialog, QRadioButton
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtProperty, QSize, Qt, QRectF, QEvent

from conponent import ClickableLabel, PercentProgressBar
from PyQt5.QtGui import QColor, QPainter, QFont, QPixmap, QImage

from conf import DESCRIPTION, CHECKPOINTS,PATH_SRC,MODEL_LIST


from utils import image2block, infer, read_image


class PredictionWorker(QObject):
    update_progress = pyqtSignal(int)

    def predict(self, model, device, image_list, film,model_name, temperature=1.0, quality=100, padding=16, patch_size=256,
                batch=8):
        # model = model.to(device)
        start = 0
        for n, image in enumerate(image_list):
            # img = cv2.imread(image) if isinstance(image, str) else image
            img = read_image(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
            target = torch.zeros((row * patch_size, col * patch_size, 3), dtype=torch.float32)
            # 第n张图片的耗时时间 均分
            end = 100 / len(image_list) * (n + 1)
            each_start = start
            with torch.no_grad():
                for i in range(0, len(split_images), batch):
                    batch_input = torch.cat(split_images[i:i + batch], dim=0)
                    batch_output = model(batch_input.to(device))
                    batch_output = batch_output[:, :, padding:-padding, padding:-padding].permute(0, 2, 3,
                                                                                                  1).cpu()
                    for j, output in enumerate(batch_output):
                        y = (i + j) // col * patch_size
                        x = (i + j) % col * patch_size
                        target[y:y + patch_size, x:x + patch_size] = output
                    if end == 100:
                        each_end = 101
                    else:
                        each_end = int(min(end, each_start + (end - start) * min(1.0, (i + 1) / len(split_images)))) + 1
                    for num in range(each_start, each_end):
                        self.update_progress.emit(num)
                        time.sleep(0.05)
                    each_start = each_end
            start = int(end)
            file_name, _ = os.path.splitext(image)
            target = target[:img.shape[0], :img.shape[1]].numpy()
            # add unnormalize
            mean = np.array([-2.12, -2.04, -1.80])
            std = np.array([4.36, 4.46, 4.44])
            target = (target - mean) / std
            target = (1.0 - temperature) * img + temperature * target
            target = np.clip(target * 255, a_min=0, a_max=255).astype(np.uint8)
            target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_name + f"_{film}_{model_name}" + '.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, quality])


class PredictionThread(QThread):
    def __init__(self, image_list, model, device, image_quality, film, temperature,model_name):
        super().__init__()
        self.worker = PredictionWorker()
        self.image_list = image_list
        self.model = model
        self.model_name = model_name
        self.device = device
        self.quality = image_quality
        self.film = film
        self.temperature = temperature

    def run(self):
        self.worker.predict(model=self.model, device=self.device,
                            image_list=self.image_list,
                            quality=self.quality,
                            film=self.film,
                            temperature=self.temperature,model_name=self.model_name)


class MyMainWindow(QMainWindow):

    def __init__(self, default_model='UNet', default_film='NC'):
        super().__init__()
        # self.gray_list = ['A']
        self.brand_name='Fuji'
        self.predict_image = None
        self.save_path = None
        self.showing_image = None  # 当前正在显示的图像
        self.film_name = default_film
        self.model_name = None
        self.quality_num = 100  # 默认图像质量
        self.temp_pred_img = None
        self.model=None
        self.checkpoints_dict = CHECKPOINTS
        self.description = DESCRIPTION
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.load_model(model_name=default_model, film_name=self.film_name)
        self.setGeometry(100, 100, 600, 500)
        self.setMinimumSize(600, 500)
        self.set_layout()
        self.set_style()

    def load_model(self, model_name, film_name):
        # 加载模型
        if self.model_name != model_name:
            self.model_name = model_name
            self.model = MODEL_LIST[model_name]['model']
            self.model = self.model.to(self.device)

        pth = torch.load(os.path.join(self.checkpoints_dict[self.brand_name][film_name], model_name.lower() + '.pth'),
                         map_location=self.device)
        self.model.load_state_dict(state_dict=pth)
        self.model.eval()
        # print(f"load model:{model_name}")

    def set_layout(self):
        """
        设置整体布局
        """
        # self.setWindowTitle("Filter Simulator")
        # 创建一个中心的 QWidget 用于容纳垂直布局
        main_window = QWidget()
        self.setCentralWidget(main_window)
        # 整体设置垂直布局->上中下
        window_layout = QVBoxLayout()
        window_layout.setSpacing(0)
        window_layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        self.title = QLabel('Film Simulate')
        window_layout.addWidget(self.title, 4)

        # 滤镜类型选择
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(0)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_fj = ClickableLabel('Fuji', self)
        brand_fj.setStyleSheet("background-color: #435f87;color: #ffffff")
        brand_kd = ClickableLabel('Kodak', self)
        brand_om = ClickableLabel('Olympus', self)
        brand_rh = ClickableLabel('Ricoh', self)
        brand_or = ClickableLabel('Other', self)
        self.brand_labels = [brand_fj, brand_kd, brand_om, brand_rh,brand_or]
        brand_layout.addWidget(brand_fj, 1)
        brand_layout.addWidget(brand_kd, 1)
        brand_layout.addWidget(brand_om, 1)
        brand_layout.addWidget(brand_rh, 1)
        brand_layout.addWidget(brand_or, 1)

        brand = QWidget()
        brand.setLayout(brand_layout)
        window_layout.addWidget(brand, 1)

        # 操作区
        operating = QWidget()
        operating_layout = QHBoxLayout()
        operating_layout.setSpacing(0)
        operating_layout.setContentsMargins(0, 0, 0, 0)
        # 滤镜选择区域
        self.film_layout = QVBoxLayout()
        self.film_area = QScrollArea()
        self.film_area.setWidgetResizable(True)
        self.film_content = QWidget()
        self.film_content.setContentsMargins(5, 5, 5, 0)
        self.create_button()  # 初始化按钮
        self.film_content.setLayout(self.film_layout)
        self.film_area.setWidget(self.film_content)
        operating_layout.addWidget(self.film_area, 2)
        # 预测区域，设置垂直布局
        prediction = QWidget()
        prediction_layout = QVBoxLayout()
        prediction_layout.setSpacing(0)
        prediction_layout.setContentsMargins(0, 0, 0, 0)
        self.img_input = QLabel('Drag/Upload Image Here')
        # self.img_input.setMouseTracking(True)
        # self.img_input.setAcceptDrops(True) # 启用 QLabel 接受拖放事件
        # self.img_input = ImageLabel('Drag Image Here')
        prediction_layout.addWidget(self.img_input, 4)
        # 按钮和进度条，设置水平布局
        self.slider = QWidget()
        slider_layout = QVBoxLayout()
        # 画质调节
        self.quality_slider = QSlider(Qt.Horizontal, self)
        self.quality_slider.setToolTip('<font color="black">Quality</font>')
        self.quality_slider.valueChanged.connect(self.set_quality)
        slider_layout.addWidget(self.quality_slider)
        # 温度系数调节
        self.temp_slider = QSlider(Qt.Horizontal, self)
        self.temp_slider.setToolTip('<font color="black">Temperature</font>')
        self.temp_slider.valueChanged.connect(self.set_temperature)
        slider_layout.addWidget(self.temp_slider)
        self.slider.setLayout(slider_layout)
        # 进度条和开始按钮
        self.button_bar = QWidget()
        button_layout = QHBoxLayout()
        self.progress_bar = PercentProgressBar(self, showFreeArea=True,
                                               backgroundColor=QColor(151, 53, 50),
                                               borderColor=QColor(0, 0, 0),
                                               borderWidth=18)
        self.start_button = QPushButton()
        self.start_button.clicked.connect(self.start_prediction)
        button_layout.addWidget(self.progress_bar)
        button_layout.addWidget(self.start_button)
        self.button_bar.setLayout(button_layout)
        slider_button_bar = QWidget()
        slider_button_bar_layout = QHBoxLayout()
        slider_button_bar_layout.addWidget(self.slider)
        slider_button_bar_layout.addWidget(self.button_bar)
        slider_button_bar_layout.setSpacing(0)
        slider_button_bar_layout.setContentsMargins(0, 0, 0, 0)
        slider_button_bar.setLayout(slider_button_bar_layout)
        prediction_layout.addWidget(slider_button_bar, 1)
        prediction.setLayout(prediction_layout)
        operating_layout.addWidget(prediction, 5)
        operating.setLayout(operating_layout)
        window_layout.addWidget(operating, 15)

        # 底部留白区域
        self.bottom = QWidget()
        self.model1 = QRadioButton("FilmCNN")
        self.model1.toggled.connect(lambda: self.load_model('FilmCNN', self.film_name))
        self.model2 = QRadioButton("UNet")
        self.model2.toggled.connect(lambda: self.load_model('UNet', self.film_name))
        if self.model_name.lower() == 'unet':
            self.model2.setChecked(True)
        else:
            self.model1.setChecked(True)

        model_group = QButtonGroup(self)
        model_group.addButton(self.model1)
        model_group.addButton(self.model2)

        # self.combo.currentIndexChanged.connect(self.switch_models)
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.model1)
        bottom_layout.addWidget(self.model2)
        # bottom_layout.setAlignment(self.model1,Qt.AlignLeft)
        # bottom_layout.setAlignment(self.model2,Qt.AlignLeft)
        self.bottom.setLayout(bottom_layout)
        window_layout.addWidget(self.bottom, 1)

        # 设置中心窗口的布局
        main_window.setLayout(window_layout)
        self.setAcceptDrops(True)
        self.warning_box = QMessageBox()

    def create_button(self):
        # 创建所需要的按钮
        self.filter_button_group = QButtonGroup()
        # 初始化fuji按钮
        self.filter_buttons = []
        film_list = sorted([i for i in os.listdir(os.path.join(PATH_SRC,self.brand_name)) if i.endswith('ORG.png')])
        for film in film_list:  # 根据图片动态创建按钮
            button = QPushButton()
            button.setFixedWidth(70)
            button.setFixedHeight(70)  # 设置按钮高度
            if 'ORG' in film:
                button.setToolTip(f'<font color="black">{self.description[self.brand_name][film.replace("-ORG.png", "")]}</font>')
            if film.replace('-ORG.png', '') == self.film_name:
                button.setStyleSheet(
                    "QPushButton { border-image: url("
                    + os.path.join(PATH_SRC,self.brand_name, film.replace('-ORG', '')).replace('\\', '/')
                    + "); }"  # 背景颜色
                )
            else:
                button.setStyleSheet(
                    "QPushButton { border-image: url("
                    + os.path.join(PATH_SRC,self.brand_name,  film).replace('\\', '/')
                    + "); }"  # 背景颜色
                )
            if not os.path.exists(
                    os.path.join(self.checkpoints_dict[self.brand_name][film.replace('-ORG.png', '')], self.model_name.lower()+'.pth')):
                button.setEnabled(False)
            self.filter_buttons.append((button, film))
            # 按钮的滤镜选择事件
            button.clicked.connect(lambda state, button=button: self.switch_film(button))
            self.filter_button_group.addButton(button)
            self.film_layout.addWidget(button)

    def set_style(self):
        """
        设置样式
        """
        # 标题区域
        font1 = QFont()
        font1.setBold(True)
        font1.setPointSize(72)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(font1)
        self.title.setStyleSheet("background-color: #435f87;"
                                 "color: #ffffff;")

        # 滑块、按钮、进度条区域
        # self.button_bar.setStyleSheet("background-color: #ffffff;")
        self.button_bar.setStyleSheet("background-color: #ffffff;")
        self.slider.setStyleSheet("background-color: #ffffff;")
        self.quality_slider.setMinimum(1)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setSingleStep(1)
        self.quality_slider.setValue(100)
        self.quality_slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            "border: 1px solid gray;"
            "height: 5px;"
            "left: 10px;"
            "right: 20px;}"
            "QSlider::handle:horizontal {"
            "border: 1px solid gray;"
            "background:white;"
            "border-radius: 7px;"
            "width: 14px;"
            "height:14px;"
            "margin: -6px;}"
            "QSlider::add-page:horizontal{background: #292a2e;}"
            "QSlider::sub-page:horizontal{background: #973532; }"
        )

        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(100)
        self.temp_slider.setSingleStep(10)
        self.temp_slider.setValue(100)
        self.temp_slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            "border: 1px solid gray;"
            "height: 5px;"
            "left: 10px;"
            "right: 20px;}"
            "QSlider::handle:horizontal {"
            "border: 1px solid gray;"
            "background:white;"
            "border-radius: 7px;"
            "width: 14px;"
            "height:14px;"
            "margin: -6px;}"
            "QSlider::add-page:horizontal{background: #292a2e;}"
            "QSlider::sub-page:horizontal{background: #508897; }"
        )

        # 图片输入框
        font2 = QFont()
        font2.setBold(True)
        font2.setPointSize(24)
        self.img_input.setStyleSheet("background-color: #292a2e;"
                                     "color: #ffffff;")
        self.img_input.setAlignment(Qt.AlignCenter)
        self.img_input.setFont(font2)

        # 按钮
        self.progress_bar.setFixedWidth(60)
        self.progress_bar.setFixedHeight(60)
        self.start_button.setFixedWidth(60)
        self.start_button.setFixedHeight(60)

        self.start_button.setStyleSheet("QPushButton { "
                                        + "border-image: url("
                                        + os.path.join(PATH_SRC, 'start.png').replace('\\',
                                                                                      '/') + ") 0 0 0 0 stretch stretch;"  # 设置背景图片的路径
                                        + "border-radius: 30px; }"  # 设置圆角半径为按钮宽度的一半
                                        + "QPushButton::hover {"
                                        + "border-image : url("
                                        + os.path.join(PATH_SRC, 'start2.png').replace('\\',
                                                                                       '/') + ") 0 0 0 0 stretch stretch;"
                                        + "border-radius: 30px; }"  # 设置圆角半径为按钮宽度的一半
                                        )
        # self.start_button.setFont(font3)

        # 滤镜选择区域
        self.film_area.setStyleSheet("background-color: #fffbf0;")
        self.film_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消滚动条
        self.film_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.film_area.setFixedWidth(106)

        # 底部区域
        self.bottom.setStyleSheet("background-color: #435f87;")

    def reflash(self, org_img, pred_img):
        # 重新渲染图像大小
        temp = self.temp_slider.value() / 100
        image = (1.0 - temp) * (org_img / 255.0) + temp * (pred_img / 255.0)
        # im = torch.clamp(im * 255, min=0, max=255).numpy().astype(np.uint8)
        image = np.clip(image * 255, a_min=0, a_max=255).astype(np.uint8)

        # 缩放目标图像自适应窗口比例大小
        w, h = self.img_input.width(), self.img_input.height()
        # image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if isinstance(image, str) else image
        # 保持原有的比例缩放
        imH, imW, _ = image.shape
        if (imW / imH) >= (w / h):
            scale_w = w
            scale_h = min(h, int(imH * (w / imW)))
        else:
            scale_w = min(w, int(imW * h / imH))
            scale_h = h
        image = cv2.resize(image, (scale_w, scale_h))
        image = QImage(image[:], image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        image = QPixmap.fromImage(image)
        self.img_input.setPixmap(image)
        self.img_input.setAlignment(Qt.AlignCenter)

    def set_quality(self):
        # 设置输出图像的质量
        self.quality_num = self.quality_slider.value()

    def set_temperature(self):
        # 动态显示图像色彩
        if self.temp_pred_img is not None:
            self.reflash(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

    def resizeEvent(self, e):
        # 改变窗口大小后QLabel中的图片重新加载
        if e.type() == QEvent.Resize:
            if self.temp_pred_img is not None:
                self.reflash(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

    def start_prediction(self):
        if self.predict_image:
            file_name, file_type = os.path.splitext(self.predict_image[-1])
            self.save_path = f"{file_name}_{self.film_name}{file_type}"
            self.prediction_thread = PredictionThread(self.predict_image, self.model, self.device,
                                                      self.quality_num, self.film_name,
                                                      self.temp_slider.value() / 100,self.model_name)
            self.prediction_thread.worker.update_progress.connect(self.update_progress_bar)
            # self.prediction_thread.finished.connect(lambda: self.finish_prediction(self.save_path))
            self.start_button.setEnabled(False)
            self.prediction_thread.start()
        else:
            # self.warning_box.setIcon(QMessageBox.Warning)
            self.warning_box.setWindowTitle("Waring")
            self.warning_box.setText("Input one image at least, Please!")
            self.warning_box.setStandardButtons(QMessageBox.Ok)
            self.warning_box.exec_()

    def update_progress_bar(self, value):

        self.progress_bar.setValue(value)
        if value == 100:
            self.start_button.setEnabled(True)
            # 新增保存路径提示
            self.warning_box.setWindowTitle("Notice！")
            self.warning_box.setText(f"Save On: {os.path.dirname(self.save_path)}")
            self.warning_box.setStandardButtons(QMessageBox.Ok)
            self.warning_box.exec_()

    def switch_brand(self, clicked_label):
        # 切换不同的滤镜品牌
        for label in self.brand_labels:
            if label == clicked_label:
                for i in reversed(range(self.film_layout.count())):
                    widget = self.film_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.deleteLater()
                self.brand_name = label.text()
                self.create_button()
                label.setStyleSheet("background-color: #435f87;"
                                    "color: #ffffff")
            else:
                label.setStyleSheet("background-color: #435f87;"
                                    "color: #292a2e;")

    def switch_film(self, clicked_button):
        # 切换不同的滤镜效果
        for button, film in self.filter_buttons:
            if button is clicked_button:
                self.film_name = film.replace('-ORG', '').replace('.png', '')
                self.load_model(self.model_name, self.film_name)
                button.setStyleSheet("QPushButton { border-image: url("
                                     + os.path.join(PATH_SRC,self.brand_name,
                                                    film.replace('-ORG', '')).replace('\\', '/')
                                     + ");}"  # 背景颜色
                                     )
                # 同时更改预览图像
                if self.showing_image is not None:
                    # img = cv2.imread(self.showing_image)
                    # img = read_image(self.showing_image)
                    img = cv2.resize(self.showing_image, (600, int(600 * self.showing_image.shape[0] / self.showing_image.shape[1])))
                    self.temp_org_img, self.temp_pred_img = infer(img, self.model, self.device)
                    self.reflash(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

            else:
                button.setStyleSheet("QPushButton { border-image: url("
                                     + os.path.join(PATH_SRC,self.brand_name, film).replace('\\', '/')
                                     + ");}"  # 背景颜色
                                     )

    def upload_image(self, mode='click', urls=None):
        if mode == 'click':
            # 打开文件对话框选择图片文件
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.tiff *tif *gif *raf "
                                      "*.cr2 *.cr3 *.arw *.rw2 *.dng *.pef *.heic)")
            file_dialog.setViewMode(QFileDialog.Detail)
            if file_dialog.exec_():
                self.predict_image = [file_dialog.selectedFiles()[0]]
        else:
            for url in urls:
                if url.isLocalFile():
                    self.predict_image = url.toLocalFile()
                    self.save_path = ''
                    # 获取所有图像
                    if os.path.isdir(self.predict_image):
                        self.predict_image = [os.path.join(self.predict_image, img) for img in
                                              os.listdir(self.predict_image)]
                    else:
                        self.predict_image = [self.predict_image]
        if self.predict_image:
            try:
                # img = cv2.imread(self.showing_image)
                self.showing_image = read_image(self.predict_image[0])
                img = cv2.resize(self.showing_image, (600, int(600 * self.showing_image.shape[0] / self.showing_image.shape[1])))
                self.temp_org_img, self.temp_pred_img = infer(img, self.model, self.device)
                self.reflash(org_img=self.temp_org_img, pred_img=self.temp_pred_img)
            except:
                self.warning_box.setWindowTitle("Waring")
                self.warning_box.setText("Unable to open this image!")
                self.warning_box.setStandardButtons(QMessageBox.Ok)
                self.warning_box.exec_()
                self.predict_image = []
                self.showing_image = None

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def mousePressEvent(self, e):
        # 限制仅点击上传区域触发点击事件
        x, y, w, h = self.film_content.width(), self.title.height(), self.img_input.width(), self.img_input.height()
        if (e.button() == Qt.LeftButton) & (e.x() >= x) & (e.x() <= x + w) & (e.y() >= y) & (e.y() <= y + h):
            self.upload_image(mode='click')

    def dropEvent(self, e):
        """
        拖拽事件：将图像拖入到选框中
        """
        self.upload_image(mode='drop', urls=e.mimeData().urls())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
