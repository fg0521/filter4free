import os.path
import cv2
import torch
import coremltools as ct
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from models import FilterSimulation2iPhone,UNet
import torch
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import subprocess




def convert(pytorch_model,model_name,size=640,image=None):
    """
    mlprogram: .mlpackage
    neuralnetwork: .mlmodel
    run `xcrun coremlcompiler compile YourModel.mlmodel ./` to convert .mlmodelc
    """
    pytorch_model.eval()
    trace_input = torch.rand(size=(1,3,size,size))
    # 追踪模型
    traced_model = torch.jit.trace(pytorch_model, trace_input)

    # 转换为 Core ML 模型
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        source='pytorch',
        inputs=[ct.ImageType(name="input",
                             shape=trace_input.shape,
                             channel_first=True,
                             color_layout=ct.colorlayout.RGB,
                             scale=1 / (255.0),
                             )]
                ,
        outputs=[ct.TensorType(name="output")],
    )
    model.author = 'Slash'
    model.version = '1.0'
    model.short_description = 'use neural network to fit camera color.'
    model_name = model_name + '.mlmodel'
    model.save(f"./mlmodel/{model_name}")

    # 输入一张图像用于验证模型
    if image is not None:
        img = cv2.imread(image)
        img = Image.fromarray(img).convert("RGB").resize((size, size))
        # ML模型
        ml_res = model.predict({"input": img})['output']
        # 使用 PyTorch 模型进行预测
        pytorch_input = transforms.ToTensor()(img).unsqueeze(0)  # 转换为张量并增加批次维度
        with torch.no_grad():
            pt_res = pytorch_model(pytorch_input).detach().cpu().numpy()
        print(f"模型验证:{np.allclose(ml_res,pt_res,1e-2)}")

    try:
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', f"./mlmodel/{model_name}", './mlmodel/'], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)
        print("Output:\n", e.output)

def predict(patch_size=624,padding=8):
    model = ct.models.MLModel('./mlmodel/CC.mlmodel')
    image = cv2.imread('/Users/maoyufeng/Downloads/63537EC1-9154-42B6-BE60-BC3C8DA6DFA1.jpeg')
    image = Image.fromarray(image).convert('RGB')
    W,H = image.size
    right_padding = padding if W % patch_size == 0 else padding + patch_size - (W % patch_size)
    bottom_padding = padding if H % patch_size == 0 else padding + patch_size - (H % patch_size)

    image = ImageOps.expand(image, border=(padding, padding, right_padding, bottom_padding), fill='white')
    # image = image.resize((W+right_padding+padding, H+bottom_padding+padding))
    new_w,new_h = image.size
    # output = Image.new('RGB', (new_w,new_h), color='white')
    output= np.empty(shape=(3,new_h,new_w))


    row = (image.size[1] - 2 * padding) // patch_size
    col = (image.size[0] - 2 * padding) // patch_size
    # 从左到右 从上到下
    patches = []
    for y1 in range(padding, row * patch_size, patch_size):
        for x1 in range(padding, col * patch_size, patch_size):
            patches.append(image.crop((x1 - padding,y1 - padding,x1 + patch_size + padding,y1 + patch_size + padding)))

    for i in tqdm(range(len(patches))):
        out = model.predict({"input":patches[i]})['output'][0]
        out = out[ :, padding:-padding, padding:-padding]
        y = i // col * patch_size
        x = i % col * patch_size
        output[:,y:y+patch_size, x:x+patch_size] = out

    output = output[:,:H,:W]
    output =  np.clip(output*255,a_min=0,a_max=255).astype(np.uint8)
    output = output.transpose((1,2,0))
    cv2.imwrite('/Users/maoyufeng/Downloads/1111/test.jpg',output,[cv2.IMWRITE_JPEG_QUALITY, 100])



if __name__ == '__main__':
    # torch_model = FilterSimulation2iPhone()
    # path = '/Users/maoyufeng/slash/project/github/filter4free/static/checkpoints'
    # name2pth = {
    #     'CC':'fuji/classic-chrome/best-v4.pth',
    #     'NC':'fuji/classic-neg/best-v4.pth',
    #     'NN':'fuji/nostalgic-neg/best-v4.pth',
    #     'Provia':'fuji/provia/best-v4.pth',
    #     'Velvia':'fuji/velvia/best-v4.pth',
    #     'Superia400':'fuji/superia400/best-v4.pth',
    #     'Pro400H':'fuji/pro400h/best-v4.pth',
    #     'ColorPlus':'kodak/colorplus/best-v4.pth',
    #     'Gold200':'kodak/gold200/best-v4.pth',
    #     'Portra400':'kodak/portra400/best-v4.pth',
    #     'Ultramax400':'kodak/ultramax400/best-v4.pth'
    # }
    # for name,pth in name2pth.items():
    #     torch_model.load_state_dict(torch.load(os.path.join(path,pth),map_location='cpu'))
    #     convert(pytorch_model=torch_model,model_name=name)
    #     print(f"Convert Model {name} Successfully...")

    predict()
