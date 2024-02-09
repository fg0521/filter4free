import copy
import os
import sys
import time
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QMainWindow, QButtonGroup, \
    QScrollArea, QPushButton, QLabel, QMessageBox, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSlider
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtProperty, QSize, Qt, QRectF, QEvent
from infer import image2block
from models import FilterSimulation
from PyQt5.QtGui import QColor, QPainter, QFont, QPixmap, QImage
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

    def predict(self, model, device, image_list, filter_name, temperature=1.0,quality=100, padding=16, patch_size=640, batch=8):
        model = model.to(device)
        channel = model.state_dict()['decoder.4.bias'].shape[0]
        start = 0
        # targets = []
        for n, image in enumerate(image_list):
            img = cv2.imread(image) if isinstance(image,str) else image
            # 对每个小块进行推理
            if channel == 1:
                # 黑白滤镜
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # 彩色滤镜
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target = np.zeros(shape=(img.shape), dtype=np.uint8)
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
                        out = out[padding:h - padding, padding:w - padding]
                        target[y:y + out.shape[0], x:x + out.shape[1], :] = out
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
            target = torch.tensor((1.0 - temperature) * (img / 255.0) + temperature * (target / 255.0))
            target = torch.clamp(target * 255, min=0, max=255).numpy().astype(np.uint8)
            target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            cv2.imwrite(file_name + f"_{filter_name}" + file_type, target, [cv2.IMWRITE_JPEG_QUALITY, quality])
            # targets.append(target)
        # return targets


class PredictionThread(QThread):
    def __init__(self, image_list, model, device, image_quality, filter_name,temperature):
        super().__init__()
        self.worker = PredictionWorker()
        self.image_list = image_list
        self.model = model
        self.device = device
        self.quality = image_quality
        self.filter_name = filter_name
        self.temperature = temperature

    def run(self):
        self.worker.predict(model=self.model, device=self.device,
                            image_list=self.image_list,
                            quality=self.quality,
                            filter_name=self.filter_name,
                            temperature=self.temperature)

def dynamic_infer(image, model, device, patch_size=448, padding=16, batch=8):
    """
    通过滑块来实现动态调整色彩
    """
    # img = cv2.imread(image)
    # channel = model.state_dict()['decoder.4.bias'].shape[0] # 获取加载的模型的通道
    model = model.to(device)
    channel = 3  # 获取加载的模型的通道
    if channel == 1:
        # 黑白滤镜
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # 彩色滤镜
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 对每个小块进行推理
    target = np.zeros(shape=(image.shape), dtype=np.uint8)
    split_images, size_list = image2block(image, patch_size=patch_size, padding=padding)
    with torch.no_grad():
        for i in range(0, len(split_images), batch):
            input = torch.vstack(split_images[i:i + batch])
            input = input.to(device)
            output = model.forward(input)
            for k in range(output.shape[0]):
                # RGB Channel
                out = torch.clamp(output[k, :, :, :] * 255, min=0, max=255).byte().permute(1, 2, 0).detach().cpu().numpy()
                x, y, w, h = size_list[i + k]
                out = cv2.resize(out, (w, h))
                out = out[padding:h - padding, padding:w - padding]
                target[y:y + out.shape[0], x:x + out.shape[1]] = out

    return image,target



class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predict_image = ''
        self.save_path = ''
        self.showing_image = ''# 当前正在显示的图像
        self.default_filter = 'FJ-V'
        self.quality_num = 100  # 默认图像质量
        # self.temperature = 100  # 默认温度系数 100为正常滤镜
        self.temp_pred_img = None
        self.model = FilterSimulation()
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.background_img = cv2.cvtColor(cv2.imread(os.path.join(file_path,'static/src/background.jpg')),cv2.COLOR_BGR2RGB)
        self.checkpoints_dict = {
            'FilmMask': os.path.join(file_path, 'static', 'checkpoints', 'film-mask', 'best.pth'),  # 去色罩
            # Fuji Filters
            'FJ-A': '',  # ACROS
            'FJ-CC': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'classic-chrome', 'best.pth'),
            # CLASSIC CHROME
            'FJ-E': '',  # ETERNA
            'FJ-EB': '',
            # ETERNA BLEACH BYPASS
            'FJ-NC': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'classic-neg', 'best.pth'),
            # CLASSIC Neg.
            'FJ-NH': '',  # PRO Neg.Hi
            'FJ-NN': os.path.join(file_path, 'static', 'checkpoints', 'fuji', 'nostalgic-neg', 'best.pth'),
            # NOSTALGIC Neg.
            'FJ-NS': '',  # PRO Neg.Std
            'FJ-S': '',  # ASTIA
            'FJ-STD': '',  # PROVIA
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
        self.description = {
            'FilmMask': '去除彩色负片的色罩',
            # from Fuji xs20
            'FJ-A': '使用丰富细节和高锐度进行黑白拍摄',
            'FJ-CC': '暗部色彩强烈，高光色彩柔和，加强阴影对比度，呈现平静的画面',
            'FJ-E': '适用于影片外观视频的柔和颜色和丰富的阴影色调',
            'FJ-EB': '低饱和度对比度的独特颜色，适用于静态图像和视频',
            'FJ-NC': '硬色调增强的色彩来增加图像深度',
            'FJ-NH': '适合对比度稍高的肖像',
            'FJ-NN': '琥珀色高光和丰富的阴影色调，用于打印照片',
            'FJ-NS': '中性色调，最适合编辑图像，适合具有柔光和渐变和肤色的人像',
            'FJ-S': '色彩对比度柔和，人像摄影',
            'FJ-STD': '适合各种拍摄对象',
            'FJ-V': '再现明亮色彩，适合自然风光摄影',
            'FJ-Pro400H': '适合各种摄影类型，特别是婚礼、时尚和生活方式摄影',
            'FJ-Superia400': '鲜艳的色彩，自然肤色，柔和的渐变',
            'FJ-C100': '较高的锐度，光滑和细腻颗粒',
            # https://asset.fujifilm.com/master/emea/files/2020-10/98c3d5087c253f51c132a5d46059f131/films_c200_datasheet_01.pdf
            'FJ-C200': '光滑、美丽、自然的肤色',
            'FJ-C400': '充满活力、逼真的色彩和细腻的颗粒',
            'FJ-Provia400X': '一流的精细粒度和清晰度，适合风景、自然、快照和人像摄影',
            # from google
            'KD-ColorPlus':'颗粒结构细、清晰度高、色彩饱和度丰富',
            'KD-Gold200': '具有色彩饱和度、细颗粒和高清晰度的出色组合',
            # from https://imaging.kodakalaris.com/sites/default/files/wysiwyg/KodakUltraMax400TechSheet-1.pdf
            'KD-UltraMax400':'明亮，充满活力的色彩，准确的肤色再现自然的人物照片',
            'KD-Portra400': '自然肤色、高细节和细颗粒，自然的暖色调',
            'KD-Portra160NC': '微妙的色彩和平滑、自然的肤色',
            'KD-E100': '极细的颗粒结构，充满活力的显色性，整体低对比度轮廓',
            'KD-Ektar100': '超鲜艳的调色、高饱和度和极其精细的颗粒结构',
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
            if 'ORG' in filter_name:
                button.setToolTip(f'{self.description[filter_name.replace("-ORG.png","")]}')
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
        self.sliders = QWidget()
        slider_layout = QVBoxLayout()
        self.quality_slider = QSlider(Qt.Horizontal, self)
        self.temp_slider = QSlider(Qt.Horizontal, self)
        self.quality_slider.setToolTip('Quality')
        self.temp_slider.setToolTip('Temperature')
        slider_layout.addWidget(self.quality_slider)
        slider_layout.addWidget(self.temp_slider)
        self.sliders.setLayout(slider_layout)

        self.button_bar = QWidget()
        button_layout = QHBoxLayout()

        self.progress_bar = PercentProgressBar(self, showFreeArea=True,
                                               backgroundColor=QColor(178, 89, 110),
                                               borderColor=QColor(118, 179, 226),
                                               borderWidth=10)
        self.start_button = QPushButton()
        self.start_button.clicked.connect(self.start_prediction)
        self.quality_slider.valueChanged.connect(self.set_quality)
        self.temp_slider.valueChanged.connect(self.set_temperature)

        button_layout.addWidget(self.progress_bar)
        button_layout.addWidget(self.start_button)
        self.button_bar.setLayout(button_layout)


        slider_button_bar = QWidget()
        slider_button_bar_layout = QHBoxLayout()
        slider_button_bar_layout.addWidget(self.sliders)
        slider_button_bar_layout.addWidget(self.button_bar)
        slider_button_bar_layout.setSpacing(0)
        slider_button_bar_layout.setContentsMargins(0, 0, 0, 0)
        slider_button_bar.setLayout(slider_button_bar_layout)



        # predict_layout.addWidget(self.button_bar, STYLE['start_button']['grid3'])
        predict_layout.addWidget(slider_button_bar, STYLE['start_button']['grid3'])

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
        # self.button_bar.setStyleSheet("background-color: #e9edf7;")
        self.button_bar.setStyleSheet("background-color: #d6def0;")
        self.sliders.setStyleSheet("background-color: #d6def0;")
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
            "QSlider::add-page:horizontal{background: #3a3c42;}"
            "QSlider::sub-page:horizontal{background: #4DA690; }"
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
        """
        拖拽事件：将图像拖入到选框中
        """
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.predict_image = url.toLocalFile()
                self.save_path = ''
                # 获取所有图像
                if os.path.isdir(self.predict_image):
                    self.predict_image = [os.path.join(self.predict_image, img) for img in os.listdir(self.predict_image)]
                else:
                    self.predict_image = [self.predict_image]
                # 判断图像类型
                self.predict_image = [img for img in self.predict_image if self.judge_image(img)]
                if self.predict_image:
                    self.showing_image = self.predict_image[0]
                    # 预测小图像
                    img = cv2.imread(self.showing_image)
                    img = cv2.resize(img,(600,int(600*img.shape[0]/img.shape[1])))
                    self.temp_org_img,self.temp_pred_img = dynamic_infer(img, self.model, self.device)
                    self.reshow(org_img=self.temp_org_img,pred_img=self.temp_pred_img)
                else:
                    self.warning_box.setWindowTitle("Waring")
                    self.warning_box.setText("Unable to open this image!")
                    self.warning_box.setStandardButtons(QMessageBox.Ok)
                    self.warning_box.exec_()
                    self.predict_image = []
                    self.showing_image = ''

    def reshow(self,org_img,pred_img):
        temp = self.temp_slider.value() / 100
        im = torch.tensor((1.0 - temp) * (org_img / 255.0) + temp * (pred_img / 255.0))
        im = torch.clamp(im * 255, min=0, max=255).numpy().astype(np.uint8)
        self.display4image(im)

    def display4image(self, image):

        w, h = self.img_input.width(), self.img_input.height()
        im = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB) if isinstance(image,str) else image
        # 保持原有的比例缩放
        imH, imW, _ = im.shape
        if (imW / imH) >= (w / h):
            scale_w = w
            scale_h = min(h, int(imH * (w / imW)))
        else:
            scale_w = min(w, int(imW * h / imH))
            scale_h = h
        im = cv2.resize(im, (scale_w, scale_h))

        img_RGB = copy.deepcopy(self.background_img)
        img_RGB = cv2.resize(img_RGB, (w, h))
        x_pad, y_pad = int((w - scale_w) / 2), int((h - scale_h) / 2)
        img_RGB[max(y_pad, 0):min(y_pad + scale_h, h), max(x_pad, 0):min(x_pad + scale_w, w), :] = im
        # img_RGB = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
        # img_RGB = show_image
        q_image = QImage(img_RGB[:], img_RGB.shape[1], img_RGB.shape[0], img_RGB.shape[1] * 3,
                         QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        # pixmap = QPixmap(image)
        # self.img_input.setPixmap(pixmap.scaled(self.img_input.size(), transformMode=Qt.SmoothTransformation))
        self.img_input.setPixmap(pixmap)
        self.img_input.setAlignment(Qt.AlignCenter)

    def judge_image(self,image):
        image_type = image.split('.')[-1]
        if image_type.lower() in ['jpg','png','jpeg','tif','tiff','jp2']:
            if cv2.imread(image) is not None:
                return True
            else:
                return False
        else:
            return False

    def set_quality(self):
        # 设置输出图像的质量
        self.quality_num = self.quality_slider.value()

    def set_temperature(self):
        # 动态显示图像色彩
        if self.temp_pred_img is not None:
            self.reshow(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

    def resizeEvent(self, e):
        # 改变窗口大小后QLabel中的图片重新加载
        if e.type() == QEvent.Resize:
            if self.temp_pred_img is not None:
                self.reshow(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

    def start_prediction(self):
        if self.predict_image:
            file_name, file_type = os.path.splitext(self.predict_image[-1])
            self.save_path = f"{file_name}_{self.default_filter}{file_type}"
            self.prediction_thread = PredictionThread(self.predict_image, self.model, self.device,
                                                      self.quality_num, self.default_filter,self.temp_slider.value()/100)
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

    # def finish_prediction(self,save_path):
    #     self.finishing_image = save_path
    #     self.display4image(self.finishing_image,dynamic=False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value == 100:
            self.start_button.setEnabled(True)
            # todo 新增保存路径提示
            self.warning_box.setWindowTitle("Notice！")
            self.warning_box.setText(f"Save Dir: {os.path.dirname(self.save_path)}")
            self.warning_box.setStandardButtons(QMessageBox.Ok)
            self.warning_box.exec_()

    def choose4filters(self, clicked_button):
        for button, filter_name in self.filter_buttons:
            if button is clicked_button:
                self.default_filter = filter_name.replace('-ORG', '').replace('.png', '')
                pth_name = self.checkpoints_dict[self.default_filter]
                if self.default_filter in self.gray_list:
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
                # 同时更改预览图像
                if self.showing_image:
                    img = cv2.imread(self.showing_image)
                    img = cv2.resize(img, (600, int(600 * img.shape[0] / img.shape[1])))
                    self.temp_org_img, self.temp_pred_img = dynamic_infer(img, self.model, self.device)
                    self.reshow(org_img=self.temp_org_img, pred_img=self.temp_pred_img)

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
