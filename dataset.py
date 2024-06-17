import cv2
import numpy as np
import skimage
from skimage.color import rgb2lab
from torch.utils.data import Dataset
import os
import torch
import numbers
from typing import Tuple, Sequence
from torchvision.transforms import functional as TF
from torchvision import transforms
from PIL import Image



class MaskDataset(Dataset):

    def __init__(self, dataset_path, mode, channel='rgb', resize=448):
        super(MaskDataset, self).__init__()
        self.mode = mode
        self.dataset_path = dataset_path
        self.data = self.get_data()
        self.channel = channel
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
            org_tensor = self.transform(Image.open(data['org_image']).convert('RGB'))
            goal_tensor = self.transform(Image.open(data['goal_image']).convert('RGB'))
        elif self.channel == 'lab':
            # resize
            # org_im = Image.open(data['org_image']).resize((self.resize, self.resize))
            # goal_im = Image.open(data['goal_image']).resize((self.resize, self.resize))
            # to numpy
            org_im = cv2.cvtColor(cv2.imread(data['org_image']), cv2.COLOR_BGR2LAB) / 255.0
            goal_im = cv2.cvtColor(cv2.imread(data['goal_image']), cv2.COLOR_BGR2LAB) / 255.0
            # RGB->LAB and softmax to [0,1]
            # org_im = np.clip((rgb2lab(org_im / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            # goal_im = np.clip((rgb2lab(goal_im / 255.0) + [0, 128, 128]) / [100, 255, 255], 0, 1)
            # to tensor
            org_tensor = torch.from_numpy(org_im).permute(2, 0, 1).to(torch.float32)
            goal_tensor = torch.from_numpy(goal_im).permute(2, 0, 1).to(torch.float32)
        elif self.channel == 'gray':
            org_tensor = self.transform(Image.open(data['org_image']).convert(mode='L'))
            goal_tensor = self.transform(Image.open(data['goal_image']).convert(mode='L'))
        elif self.channel == 'gray2rgb':
            org_tensor = self.transform(Image.open(data['org_image']).convert(mode='L'))
            goal_tensor = self.transform(Image.open(data['goal_image']).convert(mode='RGB'))
        return org_tensor, goal_tensor

    def __len__(self):
        return len(self.data)


class RandomCropThreeInstances:
    @staticmethod
    def get_params(img: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(self._setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img1, img2, img3):
        if self.padding is not None:
            img1 = TF.pad(img1, self.padding, self.fill, self.padding_mode)
            img2 = TF.pad(img2, self.padding, self.fill, self.padding_mode)
            img3 = TF.pad(img3, self.padding, self.fill, self.padding_mode)

        width, height = img1.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img1 = TF.pad(img1, padding, self.fill, self.padding_mode)
            img2 = TF.pad(img2, padding, self.fill, self.padding_mode)
            img3 = TF.pad(img3, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img1 = TF.pad(img1, padding, self.fill, self.padding_mode)
            img2 = TF.pad(img2, padding, self.fill, self.padding_mode)
            img3 = TF.pad(img2, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img1, self.size)

        return TF.crop(img1, i, j, h, w), TF.crop(img2, i, j, h, w), TF.crop(img3, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

    def _setup_size(self, size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size


class NormalizeThreeInstances(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor1, tensor2, tensor3):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return TF.normalize(tensor1, self.mean, self.std, self.inplace), \
               TF.normalize(tensor2, self.mean, self.std, self.inplace), \
               TF.normalize(tensor3, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ResizeThreeInstances(torch.nn.Module):
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def forward(self, img1, img2, img3):
        return TF.resize(img1, self.size, self.interpolation), \
               TF.resize(img2, self.size, self.interpolation), \
               TF.resize(img3, self.size, self.interpolation)

    def __repr__(self):
        _pil_interpolation_to_str = {
            Image.NEAREST: 'PIL.Image.NEAREST',
            Image.BILINEAR: 'PIL.Image.BILINEAR',
            Image.BICUBIC: 'PIL.Image.BICUBIC',
            Image.LANCZOS: 'PIL.Image.LANCZOS',
            Image.HAMMING: 'PIL.Image.HAMMING',
            Image.BOX: 'PIL.Image.BOX',
        }
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomHorizontalFlipThreeInstances:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, img3):
        if torch.rand(1) < self.p:
            return TF.hflip(img1), TF.hflip(img2), TF.hflip(img3)
        return img1, img2, img3

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor:
    def __call__(self, img1, img2, img3):
        return TF.to_tensor(img1), TF.to_tensor(img2), TF.to_tensor(img3)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, img3):
        for t in self.transforms:
            img1, img2, img3 = t(img1, img2, img3)
        return img1, img2, img3

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    img_tensor = cv2.imread('/Users/maoyufeng/Downloads/iShot_2023-09-28_13.51.36.jpg')
    img_tensor1 = 1.0 / 255 * img_tensor
    im = skimage.color.rgb2lab(img_tensor)
    im1 = skimage.color.rgb2lab(img_tensor1)
    # print(im)
    print(torch.from_numpy(im / 128))
