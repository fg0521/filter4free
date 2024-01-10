import random
import cv2
import numpy as np
import skimage
import torch
from skimage.color import rgb2lab
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MaskDataset(Dataset):

    def __init__(self, dataset_path, mode, channel='rgb', resize=448):
        super(MaskDataset, self).__init__()
        self.mode = mode
        self.dataset_path = dataset_path
        self.data = self.get_data()
        self.channel = channel
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def get_data(self):
        file = [i for i in os.listdir(os.path.join(self.dataset_path, self.mode)) if
                'org' not in i and i.endswith('jpg')]
        result = [{'goal_image': os.path.join(self.dataset_path, self.mode, img),
                   'org_image': os.path.join(self.dataset_path, self.mode, f"{img[:-4]}_org.jpg")} for img in file]
        return result

    def __getitem__(self, index):
        data = self.data[index]
        if self.channel == 'rgb':
            org_tensor = self.transform(Image.open(data['org_image']))
            goal_tensor = self.transform(Image.open(data['goal_image']))
        elif self.channel == 'lab':
            # resize
            org_im = Image.open(data['org_image']).resize((self.resize, self.resize))
            goal_im = Image.open(data['goal_image']).resize((self.resize, self.resize))
            # to numpy
            org_im = np.array(org_im)
            goal_im = np.array(goal_im)
            # RGB->LAB and softmax to [0,1]
            org_im = np.clip((rgb2lab(org_im / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            goal_im = np.clip((rgb2lab(goal_im / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            # to tensor
            org_tensor = torch.from_numpy(org_im).permute(2, 0, 1).to(torch.float32)
            goal_tensor = torch.from_numpy(goal_im).permute(2, 0, 1).to(torch.float32)
        elif self.channel =='gray':
            org_tensor = self.transform(Image.open(data['org_image']).convert(mode='L'))
            goal_tensor = self.transform(Image.open(data['goal_image']).convert(mode='L'))
        elif self.channel =='gray2rgb':
            org_tensor = self.transform(Image.open(data['org_image']).convert(mode='L'))
            goal_tensor = self.transform(Image.open(data['goal_image']).convert(mode='RGB'))
        return org_tensor, goal_tensor

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    img_tensor = cv2.imread('/Users/maoyufeng/Downloads/iShot_2023-09-28_13.51.36.jpg')
    img_tensor1 = 1.0 / 255 * img_tensor
    im = skimage.color.rgb2lab(img_tensor)
    im1 = skimage.color.rgb2lab(img_tensor1)
    # print(im)
    print(torch.from_numpy(im / 128))
