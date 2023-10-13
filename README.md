# FilterSimulation

## 介绍
使用神经网络拟合各种相机滤镜、胶片色彩

## 目录介绍
1. images：用于测试的图片路径，一级为相机名称（例：olympus），二级为滤镜名称（例：rich-color）
2. checkpoints：权重文件保存路径
3. test：记录图片训练过程变化-gif


## 模型权重
#### 胶片去色罩
1. NegativeLabPro（NLP） ✔️

#### 奥林巴斯
1. 浓郁色彩 ✔️
2. 经典黑白 ✖️


#### 模型使用
在infer.py中设置参数后一键运行
