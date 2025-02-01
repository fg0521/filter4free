import os

from models import UNet, FilterSimulation4

PATH = os.path.dirname(__file__)
PATH_FUJI = os.path.join(PATH, 'static', 'checkpoints', 'fuji')
PATH_KODAK = os.path.join(PATH, 'static', 'checkpoints', 'kodak')
PATH_RICOH = os.path.join(PATH, 'static', 'checkpoints', 'ricoh')
PATH_OLYMPUS = os.path.join(PATH, 'static', 'checkpoints', 'olympus')
PATH_OTHER = os.path.join(PATH, 'static', 'checkpoints', 'other')
PATH_SRC = os.path.join(PATH, 'static', 'src')

DESCRIPTION = {
    'Fuji': {
        # from Fuji xs20
        'A': '使用丰富细节和高锐度进行黑白拍摄',
        'CC': '暗部色彩强烈，高光色彩柔和，加强阴影对比度，呈现平静的画面',
        'E': '适用于影片外观视频的柔和颜色和丰富的阴影色调',
        'EB': '低饱和度对比度的独特颜色，适用于静态图像和视频',
        'NC': '硬色调增强的色彩来增加图像深度',
        'NH': '适合对比度稍高的肖像',
        'NN': '琥珀色高光和丰富的阴影色调，用于打印照片',
        'NS': '中性色调，最适合编辑图像，适合具有柔光和渐变和肤色的人像',
        'S': '色彩对比度柔和，人像摄影',
        'STD': '适合各种拍摄对象',
        'V': '再现明亮色彩，适合自然风光摄影',
        'Pro400H': '适合各种摄影类型，特别是婚礼、时尚和生活方式摄影',
        'Superia400': '鲜艳的色彩，自然肤色，柔和的渐变',
        'C100': '较高的锐度，光滑和细腻颗粒',
        # https://asset.fujifilm.com/master/emea/files/2020-10/98c3d5087c253f51c132a5d46059f131/films_c200_datasheet_01.pdf
        'C200': '光滑、美丽、自然的肤色',
        'C400': '充满活力、逼真的色彩和细腻的颗粒',
        'Provia400X': '一流的精细粒度和清晰度，适合风景、自然、快照和人像摄影',
        'RA': '',
    },
    'Kodak': {
        # from google
        'ColorPlus': '颗粒结构细、清晰度高、色彩饱和度丰富',
        'Gold200': '具有色彩饱和度、细颗粒和高清晰度的出色组合',
        # from https://imaging.kodakalaris.com/sites/default/files/wysiwyg/KodakUltraMax400TechSheet-1.pdf
        'UltraMax400': '明亮，充满活力的色彩，准确的肤色再现自然的人物照片',
        'Portra400': '自然肤色、高细节和细颗粒，自然的暖色调',
        'Portra160NC': '微妙的色彩和平滑、自然的肤色',
        'E100': '极细的颗粒结构，充满活力的显色性，整体低对比度轮廓',
        'Ektar100': '超鲜艳的调色、高饱和度和极其精细的颗粒结构',
    },
    'Olympus': {
        "V": '浓郁色彩',
        "SF": '柔焦',
        "SL": '柔光',
        "N": '怀旧颗粒',
        "Dim": '立体',
    },
    'Ricoh': {
        "N": '怀旧颗粒',
        "Neg": '负片',
        "Pos": '正片',
        "PN": '正片负冲',
        "S": '留银冲洗'
    },
    'Other': {'FilmMask': '去除彩色负片的色罩',
              'Polaroid': '具有艺术感的色彩',
              'Canon':'',
                'Nikon':'',
              'Sony':''
              }
}

CHECKPOINTS = {
    'Other': {
        'FilmMask': os.path.join(PATH_OTHER, 'film-mask'),  # 去色罩
        'Polaroid': os.path.join(PATH_OTHER, 'polaroid'),# Polaroid
        'Canon': os.path.join(PATH_OTHER, 'canon'),
        'Nikon': os.path.join(PATH_OTHER, 'nikon'),
        'Sony': os.path.join(PATH_OTHER, 'sony')
    },
    'Fuji': {
        # Fuji Filters
        'A': os.path.join(PATH_FUJI, 'acros'),  # ACROS
        'CC': os.path.join(PATH_FUJI, 'classic-chrome'),
        'E': os.path.join(PATH_FUJI, 'enerna'),  # ETERNA
        'EB': os.path.join(PATH_FUJI, 'eb'),  # ETERNA BLEACH BYPASS
        'NC': os.path.join(PATH_FUJI, 'classic-neg'),
        'NH': os.path.join(PATH_FUJI, 'neghi'),  # PRO Neg.Hi
        'NN': os.path.join(PATH_FUJI, 'nostalgic-neg'),
        'NS': os.path.join(PATH_FUJI, 'negstd'),  # PRO Neg.Std
        'S': os.path.join(PATH_FUJI, 'astia'),  # ASTIA
        'STD': os.path.join(PATH_FUJI, 'provia'),
        'V': os.path.join(PATH_FUJI, 'velvia'),
        'Pro400H': os.path.join(PATH_FUJI, 'pro400h'),
        'Superia400': os.path.join(PATH_FUJI, 'superia400'),
        'C100': '',
        'C200': '',
        'C400': '',
        'Provia400X': '',
        'RA':os.path.join(PATH_FUJI, 'rela'),
    },
    'Kodak': {
        # Kodak Filters
        'ColorPlus': os.path.join(PATH_KODAK, 'colorplus'),
        'Gold200': os.path.join(PATH_KODAK, 'gold200'),
        'UltraMax400': os.path.join(PATH_KODAK, 'ultramax400'),
        'Portra400': os.path.join(PATH_KODAK, 'portra400'),
        'Portra160NC': os.path.join(PATH_KODAK, 'portra160nc'),
        'E100': '',
        'Ektar100': '',
    },
    'Ricoh': {
        "N": os.path.join(PATH_RICOH, 'nostalgic'),  # 浓郁色彩
        "Neg": os.path.join(PATH_RICOH, 'neg'),  # 柔焦
        "Pos": os.path.join(PATH_RICOH, 'pos'),  # 柔光
        "PN": os.path.join(PATH_RICOH, 'pn'),  # 怀旧颗粒
        "S": os.path.join(PATH_RICOH, 'Sliver'),  # 立体
    },
    'Olympus': {
        # Olympus Filters
        "V": os.path.join(PATH_OLYMPUS, 'vivid'),  # 浓郁色彩
        "SF": os.path.join(PATH_OLYMPUS, 'softfocus'),  # 柔焦
        "SL": os.path.join(PATH_OLYMPUS, 'softlight'),  # 柔光
        "N": os.path.join(PATH_OLYMPUS, 'nostalgic'),  # 怀旧颗粒
        "Dim": os.path.join(PATH_OLYMPUS, 'dimension'),  # 立体
    },
}
MODEL_LIST = {
            'UNet': {'model': UNet()},
            'FilmCNN': {'model': FilterSimulation4()}
        }