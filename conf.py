from models import FilterSimulation, FilmMask


"""
命名方式
相机厂商-滤镜名称
字典包括：model（模型结构）、checkpoint（权重路径）、comment（说明）
"""
model_cfg = {
    'Olympus-RichColor': {'model': FilterSimulation(),
                           'checkpoint': 'checkpoints/olympus/rich-color/best.pth',
                           'comment': '奥林巴斯浓郁色彩'},
    'FilmMask': {'model': FilmMask(),
                       'checkpoint': 'checkpoints/film-mask/best.pth',
                       'comment': '去色罩'},
}
