import torch
import coremltools as ct
from PIL import Image

from models import FilterSimulation
import torch
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def convert(pytorch_model,save_path='YourModelName'):
    """
    mlprogram: .mlpackage
    neuralnetwork: .mlmodel
    """
    pytorch_model.eval()
    example_input = torch.rand(1, 3, 256, 256)
    traced_model = torch.jit.trace(torch_model, example_input)
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        # inputs=[ct.TensorType(name="input", shape=(1, 3, 256, 256))],
        inputs=[ct.TensorType(name="input", shape=(ct.RangeDim(1, 4), 3, ct.RangeDim(224, 1024), ct.RangeDim(224, 1024)))],
        outputs=[ct.TensorType(name="output")]
    )
    # image = Image.open('/Users/maoyufeng/Downloads/input.jpg')
    # image = image.resize((400, 400))
    # # 将图像转换为numpy数组并确保它是float32类型
    # image_array = np.array(image).astype('float32')
    # # 归一化图像数据到0-1之间（如果需要）
    # image_array /= 255.0
    # image_array = np.transpose(image_array, (2, 0, 1))
    # image_array = np.expand_dims(image_array, axis=0)
    # output = model.predict({"input":image_array})
    # output_image_array = output['output'].squeeze(0)  # 假设批量大小为1
    # # 如果模型输出是CHW格式，需要转换为HWC格式
    # if output_image_array.shape[0] < output_image_array.shape[1]:  # C < H，意味着是CHW格式
    #     output_image_array = np.transpose(output_image_array, (1, 2, 0))
    # # 将数据范围从[0, 1]映射回[0, 255]
    # output_image_array = (output_image_array * 255).astype(np.uint8)
    # # 创建PIL.Image对象
    # output_image = Image.fromarray(output_image_array)
    # # 如果需要，可以显示图像
    # output_image.show()

    model.save(f"./{save_path}")
    # 打印模型的每一层的名称
    for layer in model.get_spec().neuralNetwork.layers:
        print(layer.name)

if __name__ == '__main__':
    # 移除dropout层
    torch_model = FilterSimulation()
    torch_model.load_state_dict(torch.load('static/checkpoints/fuji/nostalgic-neg/best.pth',map_location='cpu'))
    convert(pytorch_model=torch_model,save_path="./mlmodel/NN.mlmodel")
