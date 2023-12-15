import os
import sys
import time
import cv2
import torch
from PIL import Image
from PyQt5.QtWidgets import QMainWindow, QButtonGroup, \
    QScrollArea, QPushButton, QLabel, QMessageBox, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtProperty, QSize, Qt, QRectF, QEvent
from infer import image2block
from models import FilterSimulation, FilmMask
from PyQt5.QtGui import QColor, QPainter, QFont, QPixmap
import json

file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, 'static', 'config.json'), 'r') as f:
    STYLE = json.load(f)[sys.platform]


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
    decimals = 2  # 进度条小数点

    def __init__(self, *args, value=0, minValue=0, maxValue=100,
                 borderWidth=8, clockwise=True, showPercent=True,
                 showFreeArea=False, showSmallCircle=False,
                 textColor=QColor(255, 255, 255),
                 borderColor=QColor(0, 255, 0),
                 backgroundColor=QColor(70, 70, 70), decimals=2, **kwargs):
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
        self._decimals = decimals

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
            # painter.drawArc(rect, (0 - arcLength) *
            #                 16, -(360 - arcLength) * 16)
            painter.drawArc(rect, int((0 - arcLength) *
                                      16), int(-(360 - arcLength) * 16))

        # 绘制当前进度圆弧
        pen.setColor(self.BorderColor)
        painter.setPen(pen)
        # painter.drawArc(rect, 0, -arcLength * 16)
        painter.drawArc(rect, 0, int(-arcLength * 16))

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
        painter.setFont(QFont('Arial', STYLE['progress_bar']['font_size']))
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

    def predict(self, model, device, image_list, filter_name, quality=100, padding=16, patch_size=640, batch=8):
        model = model.to(device)
        if list(model.state_dict().keys())[-1] == 'decoder.4.bias':
            channels = model.state_dict()['decoder.4.bias'].shape[0]
        else:
            channels = 3
        start = 0
        for n, image in enumerate(image_list):
            img = Image.open(image)
            # 对每个小块进行推理
            image_size = img.size
            if channels == 1:
                target = Image.new('L', image_size)
                img = img.convert('L')
            else:
                target = Image.new('RGB', image_size)
            split_images, size_list = image2block(img, patch_size=patch_size, padding=padding)
            # 第n张图片的耗时时间 均分
            end = 100 / len(image_list) * (n + 1)
            each_start = start
            with torch.no_grad():
                for i in range(0, len(split_images), batch):
                    input = torch.vstack(split_images[i:i + batch])
                    input = input.to(device)
                    output = model(input)
                    for k in range(output.shape[0]):
                        out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2,
                                                                                                   0).detach().cpu().numpy()
                        x, y, w, h = size_list[i + k]
                        out = cv2.resize(out, (w, h))
                        if len(out.shape) == 3:
                            out = out[padding:h - padding, padding:w - padding, :]
                        else:
                            out = out[padding:h - padding, padding:w - padding]
                        target.paste(Image.fromarray(out), (x, y))
                    if end == 100:
                        each_end = 101
                    else:
                        each_end = int(min(end, each_start + (end - start) * min(1.0, (i + 1) / len(split_images)))) + 1
                    # print(end)
                    # print(each_start + (end - start) * min(1.0, (i + 1) / len(split_images)))
                    for num in range(each_start, each_end):
                        # print(each_start,'\t',each_end)
                        self.update_progress.emit(num)
                        time.sleep(0.05)
                    each_start = each_end
            start = int(end)
            file_name, file_type = os.path.splitext(image)
            target.save(file_name + f"_{filter_name}" + file_type, quality=quality)


class PredictionThread(QThread):
    def __init__(self, image_list, model, device, image_quality, filter_name):
        super().__init__()
        self.worker = PredictionWorker()
        self.image_list = image_list
        self.model = model
        self.device = device
        self.quality = image_quality
        self.filter_name = filter_name

    def run(self):
        self.worker.predict(model=self.model, device=self.device,
                            image_list=self.image_list,
                            quality=self.quality,
                            filter_name=self.filter_name)


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predict_image = ''
        self.save_path = ''
        self.default_filter = 'FJ-V'
        self.quality_num = 100
        self.model = FilterSimulation()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.checkpoints_dict = {
            'FilmMask': os.path.join(file_path, 'static', 'checkpoints', 'film-mask', 'best.pth'),  # 去色罩
            # Fuji Filters
            'FJ-A': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'acros', 'best.pth'),  # ACROS
            'FJ-CC': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'classic-chrome', 'best.pth'),
            # CLASSIC CHROME
            'FJ-E': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'eterna', 'best.pth'),  # ETERNA
            'FJ-EB': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'eterna-bleach-bypass', 'best.pth'),
            # ETERNA BLEACH BYPASS
            'FJ-NC': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'classic-neg', 'best.pth'),
            # CLASSIC Neg.
            'FJ-NH': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'pro-neg-hi', 'best.pth'),  # PRO Neg.Hi
            'FJ-NN': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'nostalgic-neg', 'best.pth'),
            # NOSTALGIC Neg.
            'FJ-NS': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'pro-neg-std', 'best.pth'),  # PRO Neg.Std
            'FJ-S': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'astia', 'best.pth'),  # ASTIA
            'FJ-STD': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'provia', 'best.pth'),  # PROVIA
            'FJ-V': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'velvia', 'best.pth'),  # VELVIA
            'FJ-Pro400H': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'pro400h', 'best.pth'),  # VELVIA
            'FJ-Superia400': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'superia400', 'best.pth'),
            # VELVIA
            'FJ-C100': '',
            'FJ-C200': '',
            'FJ-C400': '',
            'FJ-Provia400X': '',
            # Kodak Filters
            'KD-ColorPlus': os.path.join(file_path, 'static', 'checkpoints', 'kodak', 'colorplus', 'best.pth'),
            # color plus
            'KD-Gold200': os.path.join(file_path, 'static', 'checkpoints', 'kodak', 'gold200', 'best.pth'),  # gold 200
            'KD-UltraMax400': os.path.join(file_path, 'static', 'checkpoints', 'kodak', 'ultramax400', 'best.pth'),
            # ultramax 400
            'KD-Portra400': os.path.join(file_path, 'static', 'checkpoints', 'kodak', 'portra400', 'best.pth'),
            # portra 400
            'KD-Portra160NC': os.path.join(file_path, 'static', 'checkpoints', 'kodak', 'portra160nc', 'best.pth'),
            # portra 160nc
            'KD-E100': '',
            'KD-Ektar100': '',

            # # Digital
            # # Olympus Filters
            # "OM-VIVID": os.path.join(file_path, 'static', 'checkpoints', 'olympus', 'vivid', 'best.pth'),  # 浓郁色彩
            # "OM-SoftFocus": os.path.join(file_path, 'static', 'checkpoints', 'olympus', 'soft-focus', 'best.pth'),  # 柔焦
            # "OM-SoftLight": os.path.join(file_path, 'static', 'checkpoints', 'olympus', 'soft-light', 'best.pth'),  # 柔光
            # "OM-Nostalgia": os.path.join(file_path, 'static', 'checkpoints', 'olympus', 'nostalgia', 'best.pth'),   # 怀旧颗粒
            # "OM-Stereoscopic": os.path.join(file_path, 'static', 'checkpoints', 'olympus', 'stereoscopic', 'best.pth'), # 立体
            # # Richo Filters
            # 'R-Std': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'std', 'best.pth'),  # 标准
            # 'R-Vivid': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'vivid', 'best.pth'),  # 鲜艳
            # 'R-Single': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'single', 'best.pth'),  # 单色
            # 'R-SoftSingle': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'soft-single', 'best.pth'),  # 软单色
            # 'R-StiffSingle': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'stiff-single', 'best.pth'), # 硬单色
            # 'R-ContrastSingle': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'contrastSingle', 'best.pth'),    # 高对比对黑白
            # 'R-Neg': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'neg', 'best.pth'),  # 负片
            # 'R-Pos': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'pos', 'best.pth'),  # 正片
            # 'R-Nostalgia': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'nostalgia', 'best.pth'),  # 怀旧
            # 'R-HDR': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'hdr', 'best.pth'),  # HDR
            # 'R-Pos2Neg': os.path.join(file_path, 'static', 'checkpoints', 'richo', 'pos2neg', 'best.pth'),  # 正负逆冲
            # # Canon Filters
            # 'Canon': os.path.join(file_path, 'static', 'checkpoints', 'canon', 'best.pth'),  # 佳能
            # # Sony Filters
            # 'Sony': os.path.join(file_path, 'static', 'checkpoints', 'sony', 'best.pth'),  # 索尼
            # # Nikon Filters
            # 'Nikon': os.path.join(file_path, 'static', 'checkpoints', 'nikon', 'best.pth'),  # 尼康
        }
        self.gray_list = ['FJ-A']
        self.model.load_state_dict(
            state_dict=torch.load(self.checkpoints_dict[self.default_filter], map_location=self.device))
        self.set4layout()
        self.set4stylesheet()

    def set4layout(self):
        self.setWindowTitle("Filter Simulator")

        # 创建一个中心的 QWidget 用于容纳垂直布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setGeometry(STYLE['window']['x'], STYLE['window']['y'],
                         STYLE['window']['width'], STYLE['window']['height'])
        self.setMinimumSize(STYLE['window']['width'], STYLE['window']['height'])

        # 整体设置垂直布局->上中下
        window_layout = QVBoxLayout()
        window_layout.setSpacing(0)
        window_layout.setContentsMargins(0, 0, 0, 0)

        # 上布局：设置标题为 Filter For Free
        self.title = QLabel('Filter For Free')
        window_layout.addWidget(self.title, STYLE['title']['grid1'])

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
        filter_list = sorted([i for i in os.listdir(os.path.join(file_path, 'static', 'src')) if i.endswith("ORG.png")])
        for filter_name in filter_list:  # 根据图片动态创建按钮
            button = QPushButton()
            button.setFixedWidth(STYLE['filter_button']['width'])
            button.setFixedHeight(STYLE['filter_button']['height'])  # 设置按钮高度

            if filter_name.replace('-ORG.png', '') == self.default_filter:
                button.setStyleSheet(
                    "QPushButton { border-image: url("
                    + os.path.join(file_path, 'static', 'src', filter_name.replace('-ORG', '')).replace('\\', '/')
                    + "); }"  # 背景颜色
                )
            else:
                button.setStyleSheet(
                    "QPushButton { border-image: url("
                    + os.path.join(file_path, 'static', 'src', filter_name).replace('\\', '/')
                    + "); }"  # 背景颜色
                )
            if not os.path.exists(self.checkpoints_dict[filter_name.replace('-ORG.png', '')]):
                button.setEnabled(False)
            self.filter_buttons.append((button, filter_name))
            # 按钮的滤镜选择事件
            button.clicked.connect(lambda state, button=button: self.choose4filters(button))
            self.filter_button_group.addButton(button)
            filter_layout.addWidget(button)

        all_filters.setLayout(filter_layout)
        self.filters.setWidget(all_filters)
        content_layout.addWidget(self.filters, STYLE['filter_button']['grid2'])
        # 右边为预测区域，设置垂直布局
        prediction = QWidget()
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(0)
        predict_layout.setContentsMargins(0, 0, 0, 0)

        # 图片输入框
        self.img_input = QLabel('Drag Image Here')
        # self.img_input.setAcceptDrops(True) # 启用 QLabel 接受拖放事件

        predict_layout.addWidget(self.img_input, STYLE['image_input']['grid3'])

        # 按钮和进度条，设置水平布局
        self.quality_slider = QSlider(Qt.Horizontal, self)
        self.button_bar = QWidget()
        button_layout = QHBoxLayout()

        # self.progress_bar = QProgressBar()
        self.progress_bar = PercentProgressBar(self, showFreeArea=True,
                                               backgroundColor=QColor(178, 89, 110),
                                               borderColor=QColor(118, 179, 226),
                                               borderWidth=10)
        # self.progress_bar.setFormat("%.02f %%" % value)
        # self.raw_button = QPushButton()
        self.start_button = QPushButton()
        self.start_button.clicked.connect(self.start_prediction)
        self.quality_slider.valueChanged.connect(self.set_quality)
        # self.blank = QWidget()

        button_layout.addWidget(self.quality_slider)
        button_layout.addWidget(self.progress_bar)
        # button_layout.addWidget(self.raw_button)
        button_layout.addWidget(self.start_button)

        self.button_bar.setLayout(button_layout)
        predict_layout.addWidget(self.button_bar, STYLE['start_button']['grid3'])

        prediction.setLayout(predict_layout)

        content_layout.addWidget(prediction, STYLE['image_input']['grid2'])

        filters_images.setLayout(content_layout)

        window_layout.addWidget(filters_images, STYLE['filter_button']['grid1'])

        # 底部留白区域
        self.bottom = QWidget()
        window_layout.addWidget(self.bottom, STYLE['bottom']['grid1'])

        # 设置中心窗口的布局
        central_widget.setLayout(window_layout)
        self.setAcceptDrops(True)

        self.warning_box = QMessageBox()

    def set4stylesheet(self):
        # 标题区域
        font1 = QFont()
        font1.setBold(True)
        font1.setPointSize(STYLE['title']['font_size'])
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(font1)
        self.title.setStyleSheet("background-color: #76b3e2;"
                                 "color: #e9edf7;")

        # 滑块、按钮、进度条区域
        self.button_bar.setStyleSheet("background-color: #e9edf7;")
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
            "QSlider::add-page:horizontal{background: #3a3c42;}"
            "QSlider::sub-page:horizontal{background: #b2596f; }"
        )

        # 图片输入框
        font2 = QFont()
        font2.setBold(True)
        font2.setPointSize(24)
        self.img_input.setStyleSheet("background-color: #3a3c42;"
                                     "color: #e9edf7;")
        self.img_input.setAlignment(Qt.AlignCenter)
        self.img_input.setFont(font2)

        # 按钮
        self.progress_bar.setFixedWidth(STYLE['progress_bar']['width'])
        self.progress_bar.setFixedHeight(STYLE['progress_bar']['height'])
        self.start_button.setFixedWidth(STYLE['start_button']['width'])
        self.start_button.setFixedHeight(STYLE['start_button']['height'])

        self.start_button.setStyleSheet("QPushButton { "
                                        + "border-image: url("
                                        + os.path.join(file_path, 'static', 'src', 'start.png').replace('\\',
                                                                                                        '/') + ") 0 0 0 0 stretch stretch;"  # 设置背景图片的路径
                                        + "border-radius: 30px; }"  # 设置圆角半径为按钮宽度的一半
                                        + "QPushButton::hover {"
                                        + "border-image : url("
                                        + os.path.join(file_path, 'static', 'src', 'start2.png').replace('\\',
                                                                                                         '/') + ") 0 0 0 0 stretch stretch;"
                                        + "border-radius: 30px; }"  # 设置圆角半径为按钮宽度的一半
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
                self.save_path = ''
                # todo 多图像推理
                if os.path.isdir(self.predict_image):
                    l = []
                    for img in os.listdir(self.predict_image):
                        try:
                            Image.open(os.path.join(self.predict_image, img))
                            l.append(os.path.join(self.predict_image, img))
                        except:
                            continue
                    self.predict_image = l
                    self.display4image(os.path.join(file_path, 'static', 'src', 'file_temp1.png'))
                else:
                    try:
                        Image.open(self.predict_image)
                        self.display4image(self.predict_image)
                        self.predict_image = [self.predict_image]
                    except:
                        self.warning_box.setWindowTitle("Waring")
                        self.warning_box.setText("Unable to open this image!")
                        self.warning_box.setStandardButtons(QMessageBox.Ok)
                        self.warning_box.exec_()
                        self.predict_image = []

    def display4image(self, image):
        # 显示QLabel图片
        pixmap = QPixmap(image)
        self.img_input.setPixmap(pixmap.scaled(self.img_input.size(), transformMode=Qt.SmoothTransformation))
        self.img_input.setAlignment(Qt.AlignCenter)

    def set_quality(self):
        # 设置输出图像的质量
        self.quality_num = self.quality_slider.value()

    def resizeEvent(self, e):
        # 改变窗口大小后QLabel中的图片重新加载
        if e.type() == QEvent.Resize:
            if os.path.exists(self.save_path):
                self.display4image(self.save_path)
            elif os.path.exists(self.predict_image):
                self.display4image(self.predict_image)

    def start_prediction(self):

        if self.predict_image:

            if len(self.predict_image) == 1:
                file_name, file_type = os.path.splitext(self.predict_image[0])
                self.save_path = f"{file_name}_{self.default_filter}{file_type}"
            else:
                self.save_path = os.path.join(file_path, 'static', 'src', 'file_temp2.png')
            self.prediction_thread = PredictionThread(self.predict_image, self.model, self.device,
                                                      self.quality_num, self.default_filter)
            self.prediction_thread.worker.update_progress.connect(self.update_progress_bar)
            self.prediction_thread.finished.connect(lambda: self.display4image(self.save_path))
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
                self.default_filter = filter_name.replace('-ORG', '').replace('.png', '')
                pth_name = self.checkpoints_dict[self.default_filter]
                if self.default_filter == 'FilmMask':
                    self.model = FilmMask()
                # todo add gray-channel to the list
                elif self.default_filter in self.gray_list:
                    self.model = FilterSimulation(channel=1)
                else:
                    self.model = FilterSimulation(channel=3)
                self.model.load_state_dict(
                    state_dict=torch.load(pth_name, map_location=self.device))
                button.setStyleSheet("QPushButton { border-image: url("
                                     + os.path.join(file_path, 'static', 'src',
                                                    filter_name.replace('-ORG', '')).replace('\\', '/')
                                     + ");}"  # 背景颜色
                                     )
            else:
                button.setStyleSheet("QPushButton { border-image: url("
                                     + os.path.join(file_path, 'static', 'src', filter_name).replace('\\', '/')
                                     + ");}"  # 背景颜色
                                     )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
