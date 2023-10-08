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
transform= transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MaskDataset(Dataset):
    def __init__(self, dataset_path,mode,channel='rgb',resize=512):
        super(MaskDataset, self).__init__()
        self.mode = mode
        self.dataset_path = dataset_path
        self.data = self.get_data()
        self.channel = channel
        self.resize = resize

    def get_data(self):
        file = [i for i in os.listdir(os.path.join(self.dataset_path,self.mode)) if 'mask' not in i and i.endswith('jpg')]
        result = [{ 'mask':os.path.join(self.dataset_path,self.mode,img),
                    'image':os.path.join(self.dataset_path,self.mode,f"{img[:-4]}_mask.jpg")} for img in file]
        return result

    def __getitem__(self, index):
        data = self.data[index]
        # img = cv2.imread(data['image'])
        # mask = cv2.imread(data['mask'])
        # img = rgb2lab(img)[:, :, 0]
        # mask = rgb2lab(mask)[:, :, 0]
        # img_tensor = torch.Tensor(img)[None, None, :, :]
        # mask_tensor = torch.Tensor(mask)[None, None, :, :]
        if self.channel == 'rgb':
            img_tensor = transform(Image.open(data['image']))
            mask_tensor = transform(Image.open(data['mask']))
        else:
            # cv2->RGB->LAB
            im_org = cv2.imread(data['image'])[:, :, ::-1]
            im_mask = cv2.imread(data['mask'])[:, :, ::-1]
            img_tensor = cv2.resize(im_org,(self.resize,self.resize))
            mask_tensor = cv2.resize(im_mask,(self.resize,self.resize))
            img_tensor = torch.from_numpy(skimage.color.rgb2lab(img_tensor)/128).permute(2, 0, 1).to(torch.float32)
            mask_tensor = torch.from_numpy(skimage.color.rgb2lab(mask_tensor)/128).permute(2, 0, 1).to(torch.float32)
        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    img_tensor = cv2.imread('/Users/maoyufeng/Downloads/iShot_2023-09-28_13.51.36.jpg')
    img_tensor1 = 1.0/255 *img_tensor
    im = skimage.color.rgb2lab(img_tensor)
    im1 = skimage.color.rgb2lab(img_tensor1)
    # print(im)
    print(torch.from_numpy(im/128))
