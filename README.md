## FilterSimulation

#### 1.介绍
使用神经网络拟合各种相机滤镜、胶片色彩！！！

#### 2.目录介绍
1. images：用于测试的图片路径
2. test：记录图片训练过程变化-gif
3. static：权重文件和GUI资源文件

#### 3.模型&权重
###### 负片去色罩
1. NegativeLabPro（NLP）：**static/checkpoints/film-mask/best.pth** ✅️

###### 奥林巴斯色彩模拟
1. VIVID-浓郁色彩 ：**static/checkpoints/olympus/vivid/best.pth** ✅
2. SoftFocus-柔焦 ❎
3. ️SoftLight-柔光 ❎
4. Nostalgia-怀旧颗粒 ❎
5. Stereoscopic-立体 ❎

###### 富士色彩模拟
1. ACROS ❎
2. CLASSIC CHROME：**static/checkpoints/fuji/nostalgic-neg/best.pth** ✅
3. ETERNA ❎
4. ETERNA BLEACH BYPASS ❎
5. CLASSIC Neg. ❎
6. PRO Neg.Hi ❎
7. NOSTALGIC Neg. ❎
8. PRO Neg.Std ❎
9. ASTIA ❎
10. PROVIA ❎
11. VELVIA：**static/checkpoints/fuji/velvia/best.pth** ✅

###### 理光色彩模拟
1. Std-标准 ❎
2. Vivid-鲜艳 ❎
3. Single-单色 ❎
4. SoftSingle-软单色 ❎
5. StiffSingle-硬单色 ❎
6. ContrastSingle-高对比对黑白 ❎
7. Neg-负片 ❎
8. R-Pos-正片 ❎
9. R-Nostalgia-怀旧 ❎
10. R-HDR-HDR ❎
11. R-Pos2Neg-正负逆冲 ❎

###### 索尼色彩模拟

###### 尼康色彩模拟

###### 佳能色彩模拟

###### 哈苏色彩模拟

#### 4.模型&GUI使用
###### Pycharm等解释器（适用于所有平台）

1. 配置环境：**pip install -r requirements.txt**
2. **python gui.py** 使用GUI界面运行 / **python infer.py** 使用脚本推理

###### Windows GUI

1. 百度网盘链接：链接: https://pan.baidu.com/s/1eyEdHuuOhYFyIu7uGyCohg 提取码: 3v2k 
2. 运行AIFilter.dist中的**AIFilter.exe**可执行文件

###### MacOS GUI

1.  AppleSelicon：链接: https://pan.baidu.com/s/1ofQGPB9_kPnULd7wgvswEw 提取码: wcp3 
   Intel：链接: https://pan.baidu.com/s/1MFww31KUhf6mG8QQBs_zhQ 提取码: enpi 
2. 运行**AIFilter.app** 若闪退则右键**显示包内容-Contents-MacOS-AIFilter**在终端中运行

###### GUI介绍

![](comment.jpg)
