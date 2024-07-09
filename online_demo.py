import gradio as gr
from tqdm import tqdm
from models import FilterSimulation4
from infer import image2block
import torch
from PIL import Image
import cv2


def inference(img, filter_name, batch=8, patch_size=640, padding=16):
    pth_dict = {
        'Fuji Pro400H': 'static/checkpoints/fuji/pro400h/best.pth',
        'Fuji Pro.Neg.Std': 'static/checkpoints/fuji/pro_neg_std/best.pth',
        'Fuji EternaBleachBypass': 'static/checkpoints/fuji/eterna-bleach-bypass/best.pth',
        'Fuji ClassicChrome': 'static/checkpoints/fuji/classic-chrome/best.pth',
        'Fuji Eterna': 'static/checkpoints/fuji/eterna/best.pth',
        'Fuji Provia': 'static/checkpoints/fuji/provia/best.pth',
        'Fuji Superia400': 'static/checkpoints/fuji/superia400/best.pth',
        'Fuji Acros': 'static/checkpoints/fuji/acros/best.pth',
        'Fuji Pro.Neg.Hi': 'static/checkpoints/fuji/pro-neg-hi/best.pth',
        'Fuji velvia': 'static/checkpoints/fuji/velvia/best.pth',
        'Fuji Nostalgic.Neg': 'static/checkpoints/fuji/nostalgic-neg/best.pth',
        'Fuji Classic.Neg': 'static/checkpoints/fuji/classic-neg/best.pth',
        'Fuji Astia': 'static/checkpoints/fuji/astia/best.pth',
        'Kodak Colorplus': 'static/checkpoints/kodak/colorplus/best.pth',
        'Kodak Gold200': 'static/checkpoints/kodak/gold200/best.pth',
        'Kodak Portra16NC': 'static/checkpoints/kodak/portra160nc/best.pth',
        'Kodak portra400': 'static/checkpoints/kodak/portra400/best.pth'
    }
    print(pth_dict[filter_name])
    device = torch.device('mps')
    pth = torch.load(pth_dict[filter_name], map_location=device)
    channel = pth['decoder.4.bias'].shape[0]
    model = FilterSimulation4(channel=channel)
    model.load_state_dict(pth, strict=False)
    model.to(device)
    model.eval()
    # 对每个小块进行推理
    if channel == 3:
        target = Image.new('RGB', img.size)
    else:
        img = img.convert('L')
        target = Image.new('L', img.size)
    split_images, size_list = image2block(img, patch_size=patch_size, padding=padding)
    with torch.no_grad():
        for i in tqdm(range(0, len(split_images), batch),desc=filter_name):
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
    return target


if __name__ == '__main__':
    demo = gr.Blocks()
    with demo:
        filters = gr.Dropdown(['Fuji Pro400H', 'Fuji Pro.Neg.Std', 'Fuji EternaBleachBypass',
                               'Fuji ClassicChrome', 'Fuji Eterna', 'Fuji Provia', 'Fuji Superia400',
                               'Fuji Acros', 'Fuji Pro.Neg.Hi', 'Fuji velvia', 'Fuji Nostalgic.Neg',
                               'Fuji Classic.Neg', 'Fuji Astia', 'Kodak Colorplus', 'Kodak Gold200',
                               'Kodak Portra16NC', 'Kodak portra400'], label="选择滤镜")
        gr.Interface(fn=inference, inputs=[gr.Image(type="pil"), filters], outputs=gr.Image(type="pil"))
    demo.launch(share=True)
