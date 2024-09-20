"""
Pytorch 转换为 ONNX
"""
import numpy as np
import torch
from tqdm import tqdm
from models import FilterSimulation4
import onnxruntime
from infer import unnormalize, image2block
import cv2


def convert(torch_model, checkpoint, output_path='./model.onnx',size=480,check=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch_model.load_state_dict(torch.load(checkpoint, map_location=device))
    torch_model.eval()
    input_tensor = torch.rand((1, 3, size, size))
    # 导出模型
    torch.onnx.export(
        torch_model,  # 要导出的模型
        input_tensor,  # 模型的输入
        output_path,  # 输出文件名
        # export_params=True,  # 是否导出参数
        opset_version=11,  # ONNX版本
        # do_constant_folding=True,  # 是否执行常量折叠
        input_names=['input'],  # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={  # 动态轴
            'input': {0: 'batch_size'},  # 输入的batch_size可以动态变化
            'output': {0: 'batch_size'}  # 输出的batch_size可以动态变化
        }
    )
    print(f'Convert Pytorch Model to ONNX Model Successfully, Save to {output_path}...')
    # 验证模型
    if check:
        with torch.no_grad():
            torch_out = torch_model(input_tensor.to(device)).detach().cpu().numpy()
        onnx_model = onnxruntime.InferenceSession(path_or_bytes=output_path,
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        onnx_out = onnx_model.run([onnx_model.get_outputs()[0].name], {onnx_model.get_inputs()[0].name: input_tensor.numpy().astype(np.float32)})[0]
        print(f"Model Checked:{np.allclose(torch_out,onnx_out,1e-2)}")
        print(f"l1Loss:{np.mean(np.abs(onnx_out, torch_out))}")



def infer(checkpoint, image, patch_size=448, padding=16, batch=8, channel=3):
    model = onnxruntime.InferenceSession(path_or_bytes=checkpoint,
                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
    target = np.zeros((row * patch_size, col * patch_size, channel))
    for i in tqdm(range(0, len(split_images), batch)):
        batch_input = np.vstack(split_images[i:i + batch])
        batch_output = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: batch_input})[0]
        batch_output = batch_output[:, :, padding:-padding, padding:-padding].transpose(0, 2, 3, 1)
        for j, output in enumerate(batch_output):
            y = (i + j) // col * patch_size
            x = (i + j) % col * patch_size
            target[y:y + patch_size, x:x + patch_size] = output
    target = target[:img.shape[0], :img.shape[1]]
    target = unnormalize(target)
    target = np.clip(target * 255, a_min=0, a_max=255).astype(np.uint8)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', target)
    cv2.waitKey(0)


if __name__ == "__main__":
    torch_model = FilterSimulation4()
    convert(torch_model=torch_model,
            checkpoint='/Users/maoyufeng/slash/project/filter4free/pack/static/checkpoints/fuji/classic-neg/best-v4.pth')
    infer(image='/Users/maoyufeng/Downloads/iShot_2024-08-28_17.39.31.jpg',
          checkpoint='./model.onnx',
          )
