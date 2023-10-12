from models import Olympus, FilmMask

model_cfg = {
    'olympus':{'model':Olympus(),
               'checkpoint':'checkpoints/olympus/best.pth'},
    'film_mask':{'model':FilmMask(),
               'checkpoint':'checkpoints/film_mask/best.pth'},
}