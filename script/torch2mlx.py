"""
使用apple的mlx框架训练
"""
import os
import time
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import mlx.core as mx
from mlx.nn.layers.base import Module
import mlx.nn as mlxnn
import mlx.optimizers as optim
from torch.utils.data import DataLoader, Dataset


class MLXPConv2d(Module):
    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = mlxnn.Conv2d(in_channels=self.dim_conv3, out_channels=self.dim_conv3,
                                          kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)
        init_fn = mlxnn.init.he_uniform()
        init_fn(self.partial_conv3.weight)

    def __call__(self, x):
        # for training/inference
        x1 = x[:, :, :, :self.dim_conv3]
        x2 = x[:, :, :, self.dim_conv3:]
        x1 = self.partial_conv3(x1)
        x = mx.concatenate((x1, x2), axis=-1)
        return x


class MLXModel(mlxnn.Module):
    def __init__(self, channel=3):
        super(MLXModel, self).__init__()
        self.encoder = mlxnn.Sequential(
            mlxnn.Conv2d(in_channels=channel, out_channels=32, kernel_size=3, padding=1),
            mlxnn.ReLU(),
            MLXPConv2d(32),
            mlxnn.ReLU(),
            mlxnn.AvgPool2d(kernel_size=2, stride=2),
            mlxnn.Conv2d(32, 64, kernel_size=3, padding=1),
            mlxnn.ReLU(),
            MLXPConv2d(64),
            mlxnn.ReLU(),
            mlxnn.Dropout(0.1)
        )
        self.decoder = mlxnn.Sequential(
            mlxnn.Conv2d(64, 32, kernel_size=3, padding=1),
            mlxnn.ReLU(),
            mlxnn.Conv2d(32, 32, kernel_size=3, padding=1),
            mlxnn.ReLU(),
            mlxnn.Upsample(scale_factor=2.0, mode='linear'),
            mlxnn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        init_fn = mlxnn.init.he_uniform()
        for i in [0, 5]:
            init_fn(self.encoder.layers[i].weight)
        for i in [0, 2, 5]:
            init_fn(self.decoder.layers[i].weight)

    def __call__(self, x, temp=1.0):
        # 编码器
        x1 = self.encoder(x)
        # 解码器
        x1 = self.decoder(x1)
        # 引入温度系数 来控制图像变化
        x1 = (1.0 - temp) * x + temp * x1
        return x1


class MLXLoss(Module):
    def __init__(self):
        super(MLXLoss, self).__init__()
        self.eps = 1e-4

    def __call__(self, x1, x2, model=None):
        x1 = model(x1) if model is not None else x1
        l1_loss = mlxnn.losses.l1_loss(predictions=x1, targets=x2)
        rgb1 = mx.mean(x1, axis=[1, 2], keepdims=True).squeeze()
        rgb2 = mx.mean(x2, axis=[1, 2], keepdims=True).squeeze()
        r1, g1, b1 = (rgb1[:,0] - rgb1[:,1]) ** 2, (rgb1[:,1] - rgb1[:,2]) ** 2, (rgb1[:,2] - rgb1[:,0]) ** 2
        r2, g2, b2 = (rgb2[:,0] - rgb2[:,1]) ** 2, (rgb2[:,1] - rgb2[:,2]) ** 2, (rgb2[:,2] - rgb2[:,0]) ** 2
        k1 = mx.sqrt(r1 ** 2 + g1 ** 2 + b1 ** 2 + self.eps)
        k2 = mx.sqrt(r2 ** 2 + g2 ** 2 + b2 ** 2 + self.eps)
        rgb_loss = mx.mean(mx.abs(k1 - k2))
        return l1_loss + rgb_loss


def mydataset(dataset, mode='train'):
    path = os.path.join(dataset, mode)
    images = []
    for name in os.listdir(path):
        if '_org' in name:
            images.append((os.path.join(path, name), os.path.join(path, name.replace("_org", ""))))

    return images


def batch_iterate(batch_size, images):
    if len(images) % batch_size != 0:
        images.extend(images[:batch_size - (len(images) % batch_size)])

    for i in range(0, len(images) - batch_size, batch_size):
        org_images, goal_images = [], []
        for j in range(batch_size):
            # print(i * batch_size + j)
            org_img = cv2.cvtColor(cv2.imread(images[i + j][0]), cv2.COLOR_BGR2RGB)
            # org_img = Image.open(images[i + j][0]).resize((448, 448))
            org_img = mx.array(org_img) / 255.0
            # goal_img = Image.open(images[i + j][1]).resize((448, 448))
            goal_img = cv2.cvtColor(cv2.imread(images[i + j][1]), cv2.COLOR_BGR2RGB)
            goal_img = mx.array(goal_img) / 255.0
            org_images.append(org_img)
            goal_images.append(goal_img)
        yield mx.stack(org_images), mx.stack(goal_images)



def image2block(image, patch_size=448, padding=16):
    patches, size_list = [], []
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                               borderType=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101
    H, W, C = image.shape
    for x1 in range(padding, W - 2 * padding, patch_size):
        for y1 in range(padding, H - 2 * padding, patch_size):
            x2 = min(x1 + patch_size + padding, W)
            y2 = min(y1 + patch_size + padding, H)
            patch = image[y1 - padding:y2, x1 - padding:x2, :]
            size_list.append((x1 - padding, y1 - padding, patch.shape[1], patch.shape[0]))  # x,y,w,h
            patch = cv2.resize(patch, (patch_size, patch_size)) / 255.0
            if len(patch.shape) == 2:
                patch = np.array([patch])
            patches.append(patch)
    return patches, size_list



def infer(image, patch_size=448, batch=8, padding=16):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches, size_list = image2block(img, patch_size=patch_size, padding=padding)
    target = np.empty(shape=img.shape,dtype=np.uint8)
    for i in tqdm(range(0, len(patches), batch),ncols=50):
        batch_input = mx.array(np.stack(patches[i:i + batch]))
        batch_output = model(batch_input)
        batch_output = np.array(batch_output*255.0,dtype=np.uint8)
        for j, out in enumerate(batch_output):
            x, y, w, h = size_list[i + j]
            out = cv2.resize(out, (w, h))
            out = out[padding:h - padding, padding:w - padding]
            target[y:y + out.shape[0], x:x + out.shape[1]] = out
    target = np.clip(target, a_min=0, a_max=255)
    return cv2.cvtColor(target, cv2.COLOR_RGB2BGR)


def train(model,dataset='/Users/maoyufeng/slash/dataset/train_dataset/classic-neg',pretrain_model='mlx/best.npz'):
    model.load_weights(pretrain_model)
    model.eval()
    loss_fn = MLXLoss()
    train_images = mydataset(dataset, mode='train')
    val_images = mydataset(dataset, mode='val')
    mx.eval(model.parameters())
    loss_and_grad_fn = mlxnn.value_and_grad(model, loss_fn)
    grad_fn = mx.grad(loss_fn)
    lr_schedule = optim.step_decay(1e-4, 0.9, 200)
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    epoch = 50
    batch_size = 8
    max_loss = 999.0
    pbar = tqdm(total=len(train_images) // batch_size * epoch, desc="Training Progress:", ncols=80)
    for e in range(epoch):
        pbar.set_description(f"Epoch: {e + 1}")
        train_loss, val_loss = [], []
        model.train()
        for org_img, goal_img in batch_iterate(batch_size, train_images):
            loss, grads = loss_and_grad_fn(org_img, goal_img, model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss.append(loss.item())
            pbar.update(1)
            pbar.set_postfix(**{'loss': round(sum(train_loss) / len(train_loss), 4)})  # 参数列表
        model.save_weights(f'mlx/epoch{e}.npz')

        model.eval()
        for org_img, goal_img in batch_iterate(batch_size, val_images):
            loss, grads = loss_and_grad_fn(org_img, goal_img, model)
            val_loss.append(loss.item())
        print(f"Eval Loss:{round(sum(val_loss) / len(val_loss), 4)}")
        if max_loss >= sum(val_loss) / len(val_loss):
            max_loss = sum(val_loss) / len(val_loss)
            model.save_weights(f"mlx/best.npz")


if __name__ == '__main__':
    mx.set_default_device(mx.gpu)
    model = MLXModel()
    train(model)

    # model = MLXModel()
    # model.load_weights('mlx/best.npz')
    # model.eval()
    # image = Image.open('/Users/maoyufeng/slash/dataset/train_dataset/classic-neg/val/170649484824034_org.jpg')
    # image = np.array(image)
    # image = mx.array(np.array([image / 255.0]))

    # stime = time.time()
    # target = infer('/Users/maoyufeng/slash/dataset/org_dataset/classic-neg/DSCF5792_org.jpg')
    # print(time.time()-stime)
    # cv2.imwrite('/Users/maoyufeng/Downloads/test.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, 100])

    #
    # im = cv2.imread('/Users/maoyufeng/Downloads/iShot_2024-05-09_17.45.27.jpg')
    # print(im)