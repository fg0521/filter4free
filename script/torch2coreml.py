"""
Pytorch 转换为 CoreML
"""
import os.path
import cv2
import coremltools as ct
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from infer import image2block, unnormalize
from models import FilterSimulation4
import torch
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import subprocess


def convert(torch_model,checkpoint,size=480,output_path='./model.mlmodel',check=True,
            version='1.0',author = 'Slash',desc='use neural network to fit camera color.'):
    """
    mlprogram: .mlpackage
    neuralnetwork: .mlmodel
    run `xcrun coremlcompiler compile YourModel.mlmodel ./` to convert .mlmodelc
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch_model.load_state_dict(torch.load(checkpoint,map_location=device))
    torch_model.eval()
    input_tensor = torch.rand(size=(1,3,size,size))
    # 追踪模型
    traced_model = torch.jit.trace(torch_model, input_tensor)
    # 转换为 Core ML 模型
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        source='pytorch',
        inputs=[ct.ImageType(name="input",
                             shape=input_tensor.shape,
                             channel_first=True,
                             color_layout=ct.colorlayout.RGB,
                             scale=1 / (255.0),
                             )]
                ,
        outputs=[ct.TensorType(name="output")],
    )
    model.author = author
    model.version = version
    model.short_description = desc
    model.save(output_path)
    # 验证模型
    if check:
        to_pil = ToPILImage()
        # 转换为PIL Image
        img = to_pil(input_tensor.squeeze(0))
        # ML模型
        ml_out = model.predict({"input": img})['output']
        with torch.no_grad():
            torch_out = torch_model(input_tensor).detach().cpu().numpy()
        print(f"Model Checked:{np.allclose(ml_out,torch_out,0.05)}")
        print(f"l1Loss:{np.mean(np.abs(ml_out,torch_out))}")
    try:
        # 编译模型
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', output_path, os.path.dirname(output_path)], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with return code", e.returncode)
        print("Output:\n", e.output)


def infer(checkpoint, image, patch_size=448, padding=16, batch=1, channel=3):
    model = ct.models.MLModel(checkpoint)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
    target = np.zeros((row * patch_size, col * patch_size, channel))
    for i in tqdm(range(0, len(split_images), batch)):
        batch_input = unnormalize(split_images[i].squeeze().transpose(1,2,0))
        batch_input = np.clip(batch_input * 255, a_min=0, a_max=255).astype(np.uint8)
        batch_input = Image.fromarray(batch_input)
        batch_output = model.predict({"input":batch_input})['output'][0]    # 输入为Image.image格式
        batch_output = batch_output[:, padding:-padding, padding:-padding]
        y = i // col * patch_size
        x = i % col * patch_size
        batch_output = batch_output.transpose(1,2,0)
        target[y:y + patch_size, x:x + patch_size] = batch_output
    target = target[:img.shape[0], :img.shape[1]]
    target = np.clip(target * 255, a_min=0, a_max=255).astype(np.uint8)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', target)
    cv2.waitKey(0)

if __name__ == '__main__':
    torch_model = FilterSimulation4()
    convert(torch_model=torch_model, checkpoint='/Users/maoyufeng/slash/project/github/filter4free/static/checkpoints/fuji/classic-chrome/best-v4.pth')
    infer(checkpoint='./model.mlmodel',image='/Users/maoyufeng/Downloads/iShot_2024-08-28_17.39.31.jpg')
