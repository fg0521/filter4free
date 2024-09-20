"""
torch -> onnx -> tf
"""
import cv2
import keras
import numpy as np
import onnx
import onnxruntime
from onnx_tf.backend import prepare
import tensorflow as tf
from tqdm import tqdm

from infer import image2block, unnormalize


def convert(checkpoint,output_path='./tf_model',size=480,check=True):
    onnx_model = onnx.load(checkpoint)
    p = prepare(onnx_model)
    p.export_graph(output_path)
    if check:
        tf_model = tf.saved_model.load(output_path)
        input_data = tf.random.uniform([1, 3, size, size])
        infer = tf_model.signatures["serving_default"]
        tf_out = infer(input_data)['output'].numpy()
        onnx_model = onnxruntime.InferenceSession(path_or_bytes=checkpoint,
                                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        onnx_out = onnx_model.run([onnx_model.get_outputs()[0].name], {onnx_model.get_inputs()[0].name: input_data.numpy().astype(np.float32)})[0]
        print(f"Model Checked:{np.allclose(tf_out,onnx_out,1e-2)}")
        print(f"l1Loss:{np.mean(np.abs(onnx_out, tf_out))}")


def infer(checkpoint, image, patch_size=448, padding=16, batch=8, channel=3):
    model = tf.saved_model.load(checkpoint)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    split_images, row, col = image2block(img, patch_size=patch_size, padding=padding)
    target = np.zeros((row * patch_size, col * patch_size, channel))
    for i in tqdm(range(0, len(split_images), batch)):
        batch_input = np.vstack(split_images[i:i + batch])
        batch_input = tf.convert_to_tensor(batch_input)
        infer = model.signatures["serving_default"]
        batch_output = infer(batch_input)['output'].numpy()
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


if __name__ == '__main__':
    convert(checkpoint="./model.onnx")
    infer(checkpoint='tf_model',image='/Users/maoyufeng/Downloads/iShot_2024-08-28_17.39.31.jpg')