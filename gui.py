import os
import sys
import threading
import time

import cv2
import torch
from PIL import Image, ImageEnhance
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QButtonGroup, QRadioButton, \
    QScrollArea, QPushButton, QListWidget, QListWidgetItem, QLabel, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from functools import partial

from infer import image2block
from models import FilterSimulation

from PyQt5.QtCore import pyqtProperty, QSize, Qt, QRectF, QTimer
from PyQt5.QtGui import QColor, QPainter, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider


class PercentProgressBar(QWidget):
    MinValue = 0
    MaxValue = 100
    Value = 0
    BorderWidth = 8
    Clockwise = True  # 顺时针还是逆时针
    ShowPercent = True  # 是否显示百分比
    ShowFreeArea = False  # 显示背后剩余
    ShowSmallCircle = False  # 显示带头的小圆圈
    TextColor = QColor(255, 255, 255)  # 文字颜色
    BorderColor = QColor(24, 189, 155)  # 边框圆圈颜色
    BackgroundColor = QColor(70, 70, 70)  # 背景颜色

    def __init__(self, *args, value=0, minValue=0, maxValue=100,
                 borderWidth=8, clockwise=True, showPercent=True,
                 showFreeArea=False, showSmallCircle=False,
                 textColor=QColor(255, 255, 255),
                 borderColor=QColor(0, 255, 0),
                 backgroundColor=QColor(70, 70, 70), **kwargs):
        super(PercentProgressBar, self).__init__(*args, **kwargs)
        self.Value = value
        self.MinValue = minValue
        self.MaxValue = maxValue
        self.BorderWidth = borderWidth
        self.Clockwise = clockwise
        self.ShowPercent = showPercent
        self.ShowFreeArea = showFreeArea
        self.ShowSmallCircle = showSmallCircle
        self.TextColor = textColor
        self.BorderColor = borderColor
        self.BackgroundColor = backgroundColor

    def setRange(self, minValue: int, maxValue: int):
        if minValue >= maxValue:  # 最小值>=最大值
            return
        self.MinValue = minValue
        self.MaxValue = maxValue
        self.update()

    def paintEvent(self, event):
        super(PercentProgressBar, self).paintEvent(event)
        width = self.width()
        height = self.height()
        side = min(width, height)

        painter = QPainter(self)
        # 反锯齿
        painter.setRenderHints(QPainter.Antialiasing |
                               QPainter.TextAntialiasing)
        # 坐标中心为中间点
        painter.translate(width / 2, height / 2)
        # 按照100x100缩放
        painter.scale(side / 100.0, side / 100.0)

        # 绘制中心园
        self._drawCircle(painter, 50)
        # 绘制圆弧
        self._drawArc(painter, 50 - self.BorderWidth / 2)
        # 绘制文字
        self._drawText(painter, 50)

    def _drawCircle(self, painter: QPainter, radius: int):
        # 绘制中心园
        radius = radius - self.BorderWidth
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.BackgroundColor)
        painter.drawEllipse(QRectF(-radius, -radius, radius * 2, radius * 2))
        painter.restore()

    def _drawArc(self, painter: QPainter, radius: int):
        # 绘制圆弧
        painter.save()
        painter.setBrush(Qt.NoBrush)
        # 修改画笔
        pen = painter.pen()
        pen.setWidthF(self.BorderWidth)
        pen.setCapStyle(Qt.RoundCap)

        arcLength = 360.0 / (self.MaxValue - self.MinValue) * self.Value
        rect = QRectF(-radius, -radius, radius * 2, radius * 2)

        if not self.Clockwise:
            # 逆时针
            arcLength = -arcLength

        # 绘制剩余进度圆弧
        if self.ShowFreeArea:
            acolor = self.BorderColor.toRgb()
            acolor.setAlphaF(0.2)
            pen.setColor(acolor)
            painter.setPen(pen)
            painter.drawArc(rect, (0 - arcLength) *
                            16, -(360 - arcLength) * 16)

        # 绘制当前进度圆弧
        pen.setColor(self.BorderColor)
        painter.setPen(pen)
        painter.drawArc(rect, 0, -arcLength * 16)

        # 绘制进度圆弧前面的小圆
        if self.ShowSmallCircle:
            offset = radius - self.BorderWidth + 1
            radius = self.BorderWidth / 2 - 1
            painter.rotate(-90)
            circleRect = QRectF(-radius, radius + offset,
                                radius * 2, radius * 2)
            painter.rotate(arcLength)
            painter.drawEllipse(circleRect)

        painter.restore()

    def _drawText(self, painter: QPainter, radius: int):
        # 绘制文字
        painter.save()
        painter.setPen(self.TextColor)
        painter.setFont(QFont('Arial', 25))
        strValue = '{}%'.format(int(self.Value / (self.MaxValue - self.MinValue)
                                    * 100)) if self.ShowPercent else str(self.Value)
        painter.drawText(QRectF(-radius, -radius, radius * 2,
                                radius * 2), Qt.AlignCenter, strValue)
        painter.restore()

    @pyqtProperty(int)
    def minValue(self) -> int:
        return self.MinValue

    @minValue.setter
    def minValue(self, minValue: int):
        if self.MinValue != minValue:
            self.MinValue = minValue
            self.update()

    @pyqtProperty(int)
    def maxValue(self) -> int:
        return self.MaxValue

    @maxValue.setter
    def maxValue(self, maxValue: int):
        if self.MaxValue != maxValue:
            self.MaxValue = maxValue
            self.update()

    @pyqtProperty(int)
    def value(self) -> int:
        return self.Value

    @value.setter
    def value(self, value: int):
        if self.Value != value:
            self.Value = value
            self.update()

    @pyqtProperty(float)
    def borderWidth(self) -> float:
        return self.BorderWidth

    @borderWidth.setter
    def borderWidth(self, borderWidth: float):
        if self.BorderWidth != borderWidth:
            self.BorderWidth = borderWidth
            self.update()

    @pyqtProperty(bool)
    def clockwise(self) -> bool:
        return self.Clockwise

    @clockwise.setter
    def clockwise(self, clockwise: bool):
        if self.Clockwise != clockwise:
            self.Clockwise = clockwise
            self.update()

    @pyqtProperty(bool)
    def showPercent(self) -> bool:
        return self.ShowPercent

    @showPercent.setter
    def showPercent(self, showPercent: bool):
        if self.ShowPercent != showPercent:
            self.ShowPercent = showPercent
            self.update()

    @pyqtProperty(bool)
    def showFreeArea(self) -> bool:
        return self.ShowFreeArea

    @showFreeArea.setter
    def showFreeArea(self, showFreeArea: bool):
        if self.ShowFreeArea != showFreeArea:
            self.ShowFreeArea = showFreeArea
            self.update()

    @pyqtProperty(bool)
    def showSmallCircle(self) -> bool:
        return self.ShowSmallCircle

    @showSmallCircle.setter
    def showSmallCircle(self, showSmallCircle: bool):
        if self.ShowSmallCircle != showSmallCircle:
            self.ShowSmallCircle = showSmallCircle
            self.update()

    @pyqtProperty(QColor)
    def textColor(self) -> QColor:
        return self.TextColor

    @textColor.setter
    def textColor(self, textColor: QColor):
        if self.TextColor != textColor:
            self.TextColor = textColor
            self.update()

    @pyqtProperty(QColor)
    def borderColor(self) -> QColor:
        return self.BorderColor

    @borderColor.setter
    def borderColor(self, borderColor: QColor):
        if self.BorderColor != borderColor:
            self.BorderColor = borderColor
            self.update()

    @pyqtProperty(QColor)
    def backgroundColor(self) -> QColor:
        return self.BackgroundColor

    @backgroundColor.setter
    def backgroundColor(self, backgroundColor: QColor):
        if self.BackgroundColor != backgroundColor:
            self.BackgroundColor = backgroundColor
            self.update()

    def setValue(self, value):
        self.value = value

    def sizeHint(self) -> QSize:
        return QSize(100, 100)


class PredictionWorker(QObject):
    update_progress = pyqtSignal(int)

    def predict(self, image, pth):
        self.model = FilterSimulation()
        self.device = torch.device('cpu')
        # print(f'开始预测：{image}')
        self.model.load_state_dict(
            torch.load(pth, map_location=self.device))
        self.model.to(self.device)
        img = Image.open(image)
        # 对每个小块进行推理
        image_size = img.size
        num_cols = image_size[1] // 512 + 1
        target = Image.new('RGB', image_size)
        split_images, size_list = image2block(img, patch_size=512, padding=10)

        t = min(100 / len(split_images) * 8, 100)
        start = 0
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
                end = int(t * cnt)
                for num in range(start + 1, end + 1):
                    self.update_progress.emit(num)
                    time.sleep(0.1)
                start = end
                cnt += 1

        save_path = os.path.dirname(image)
        name = '.'.join(os.path.basename(image).split('.')[:-1]) + '_predict.jpg'
        target.save(os.path.join(save_path, name))
        # print(f'保存到：{os.path.join(save_path, name)}')


class PredictionThread(QThread):
    def __init__(self, image, pth):
        super().__init__()
        self.worker = PredictionWorker()
        self.image = image
        self.pth = pth

    def run(self):
        self.worker.predict(self.image, self.pth)


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predict_image = ''
        self.default_filter = 'VIVID'
        self.checkpoints_dict = {
            "VIVID": 'checkpoints/olympus/vivid/best.pth',
            'A': '',
            'CC': '',
            'E': '',
            'EB': '',
            'NC': '',
            'NH': '',
            'NN': 'checkpoints/fuji/velvia/best.pth',
            'NS': '',
            'S': '',
            'STD': '',
            'V': 'checkpoints/fuji/velvia/best.pth'
        }
        self.pth_name = self.checkpoints_dict[self.default_filter]
        self.set4layout()
        self.set4stylesheet()

    def set4layout(self):
        self.setWindowTitle("Filter Simulator")

        # 创建一个中心的 QWidget 用于容纳垂直布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setGeometry(100, 100, 600, 500)
        self.setMinimumSize(600, 500)

        # 整体设置垂直布局->上中下
        total_layout = QVBoxLayout()
        total_layout.setSpacing(0)
        total_layout.setContentsMargins(0, 0, 0, 0)

        # 上布局：设置标题为 Filter For Free
        self.title = QLabel('Filter For Free')
        total_layout.addWidget(self.title, 5)

        # 中布局：设置滤镜选择、图片输入、按钮、进度条
        # 中布局设置水平布局
        filters_images = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 左边为滤镜选择区域
        filter_layout = QVBoxLayout()
        self.filters = QScrollArea()

        all_filters = QWidget()
        all_filters.setContentsMargins(5, 5, 5, 0)
        self.filter_button_group = QButtonGroup()
        # 存放按钮和对应的图片路径
        self.filter_buttons = []
        filter_list = [i for i in os.listdir('src') if i.endswith("ORG.PNG")]
        for filter_name in filter_list:  # 根据图片动态创建按钮
            button = QPushButton()
            button.setFixedHeight(80)  # 设置按钮高度
            button.setFixedWidth(80)
            if filter_name.replace('_ORG.PNG', '') == self.default_filter:
                button.setStyleSheet(
                    "QPushButton {"
                    f"   border-image: url({os.path.join('src', filter_name.replace('_ORG', ''))});"  # 背景颜色
                    "}"
                )
            else:
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

        all_filters.setLayout(filter_layout)
        self.filters.setWidget(all_filters)
        content_layout.addWidget(self.filters, 2)
        # 右边为预测区域，设置垂直布局
        prediction = QWidget()
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(0)
        predict_layout.setContentsMargins(0, 0, 0, 0)

        # 图片输入框
        self.img_input = QLabel('Drag Image Here')
        # self.img_input.setAcceptDrops(True) # 启用 QLabel 接受拖放事件

        predict_layout.addWidget(self.img_input, 4)

        # 按钮和进度条，设置水平布局
        self.button_bar = QWidget()
        button_layout = QHBoxLayout()

        # self.progress_bar = QProgressBar()
        self.progress_bar = PercentProgressBar(self, showFreeArea=True,
                                               backgroundColor=QColor(178, 89, 110),
                                               borderColor=QColor(118, 179, 226),
                                               borderWidth=10)
        # self.raw_button = QPushButton()
        self.start_button = QPushButton()
        self.start_button.clicked.connect(self.start_prediction)
        self.blank = QWidget()

        button_layout.addWidget(self.blank)
        button_layout.addWidget(self.progress_bar)
        # button_layout.addWidget(self.raw_button)
        button_layout.addWidget(self.start_button)

        self.button_bar.setLayout(button_layout)
        predict_layout.addWidget(self.button_bar, 1)

        prediction.setLayout(predict_layout)

        content_layout.addWidget(prediction, 5)

        filters_images.setLayout(content_layout)

        total_layout.addWidget(filters_images, 15)

        # 底部留白区域
        self.bottom = QWidget()
        total_layout.addWidget(self.bottom, 1)

        # 设置中心窗口的布局
        central_widget.setLayout(total_layout)
        self.setAcceptDrops(True)

        self.warning_box = QMessageBox()

    def set4stylesheet(self):
        # 标题区域
        font1 = QFont()
        font1.setBold(True)
        font1.setPointSize(74)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(font1)
        self.title.setStyleSheet("background-color: #76b3e2;"
                                 "color: #e9edf7;")

        # 按钮、进度条区域
        self.button_bar.setStyleSheet("background-color: #e9edf7;")

        # 图片输入框
        font2 = QFont()
        font2.setBold(True)
        font2.setPointSize(24)
        self.img_input.setStyleSheet("background-color: #3a3c42;"
                                     "color: #e9edf7;")
        self.img_input.setAlignment(Qt.AlignCenter)
        self.img_input.setFont(font2)

        # 按钮
        self.progress_bar.setFixedWidth(60)
        self.progress_bar.setFixedHeight(60)
        self.start_button.setFixedHeight(60)
        self.start_button.setFixedWidth(60)
        self.start_button.setStyleSheet("QPushButton {"
                                        "border-image: url(src/start.png) 0 0 0 0 stretch stretch;"  # 设置背景图片的路径
                                        f"border-radius: 30px;"  # 设置圆角半径为按钮宽度的一半
                                        "}"
                                        "QPushButton::hover"
                                        "{"
                                        "border-image : url(src/start2.png) 0 0 0 0 stretch stretch;"
                                        f"border-radius: 30px;"  # 设置圆角半径为按钮宽度的一半
                                        "}"
                                        )
        # self.start_button.setFont(font3)

        # 滤镜选择区域
        self.filters.setStyleSheet("background-color: #efd4a7;")
        self.filters.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 取消滚动条
        self.filters.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.filters.setContentsMargins(20, 20, 20, 20)
        self.filters.setFixedWidth(118)

        # 底部区域
        self.bottom.setStyleSheet("background-color: #76b3e2;")

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

    def display4image(self, image):
        # print(f'显示：{image}')
        pixmap = QPixmap(image)
        self.img_input.setPixmap(pixmap.scaled(self.img_input.size(), transformMode=Qt.SmoothTransformation))
        self.img_input.setAlignment(Qt.AlignCenter)
        # 开始预测函数

    def start_prediction(self):
        if self.predict_image:
            save_path = os.path.dirname(self.predict_image)
            name = '.'.join(os.path.basename(self.predict_image).split('.')[:-1]) + '_predict.jpg'
            self.prediction_thread = PredictionThread(self.predict_image, self.pth_name)
            self.prediction_thread.worker.update_progress.connect(self.update_progress_bar)
            self.prediction_thread.finished.connect(lambda: self.display4image(os.path.join(save_path, name)))
            self.start_button.setEnabled(False)
            self.prediction_thread.start()
        else:
            # self.warning_box.setIcon(QMessageBox.Warning)
            self.warning_box.setWindowTitle("Waring")
            self.warning_box.setText("One Image At Least, Please!")
            self.warning_box.setStandardButtons(QMessageBox.Ok)
            self.warning_box.exec_()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.start_button.setEnabled(True)

    def choose4filters(self, clicked_button):
        for button, filter_name in self.filter_buttons:
            if button is clicked_button:
                print(filter_name)
                self.pth_name = self.checkpoints_dict[filter_name.replace('_ORG', '').replace('.PNG', '')]
                print(self.pth_name)
                button.setStyleSheet("QPushButton {"
                                     f"   border-image: url({os.path.join('src', filter_name.replace('_ORG', ''))});"  # 背景颜色
                                     "}")
            else:
                button.setStyleSheet("QPushButton {"
                                     f"   border-image: url({os.path.join('src', filter_name)});"  # 背景颜色
                                     "}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
