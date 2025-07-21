import numpy as np 
import cv2
from PIL import Image, ImageFile
import os
import pdb
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import math
import random
import augment_ops
# ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)

def add_gaussian_noise(image, mean=0, std=0.01):
    std = random.uniform(0, 10)
    image = np.array(image)
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

'''random erasing'''
def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

class FGB(data.Dataset):
    def __init__(self, txt, is_training):
        f = open(txt,'r')
        self.datalist = f.readlines()
        self.is_training = is_training
        self.resize = transforms.Resize(size=448)
        self.horizontalflip = transforms.RandomHorizontalFlip()
        self.verticalflip = transforms.RandomVerticalFlip()
        self.colorjitter = transforms.ColorJitter()
        self.randomcrop = transforms.RandomCrop((448))
        self.centercrop = transforms.CenterCrop((448))
        self.totensor = transforms.ToTensor()
        self.aa_params = dict(
                            translate_const=int(448 * 0.45),
                            img_mean=tuple([round(x * 255) for x in IMAGENET_DEFAULT_MEAN]),
                            interpolation=Image.BILINEAR
                        )
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.colorjitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.randomerasing = RandomErasing(probability=0.15, mode='pixel', max_count=1, num_splits=0, device='cpu')
        self.affine = transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1))
        self.random_aug = augment_ops.rand_augment_transform("rand-m9-mstd0.5", self.aa_params)

    def __getitem__(self, index):
        label = int(self.datalist[index].strip().split('xxxx')[-1])
        image_path_now = self.datalist[index].strip().split('xxxx')[0]

        # imgname = os.path.join(self.image_path, image_path_now)
        imgname = image_path_now
        img = Image.open(imgname)

        img = self.resize(img)
        pp = np.random.random()
        pp2 = np.random.random()
        if self.is_training:
            if pp < 2:
                img = self.randomcrop(img)
                img = img.resize((448, 448))
                img = self.random_aug(img)
            else:
                img = img.resize((448, 448))
            if pp2 < 0.5:
                img = add_gaussian_noise(img)
            img = self.verticalflip(img)
            img = self.horizontalflip(img)
        else:
            img = self.centercrop(img)
            img = img.resize((448, 448))
        img = np.array(img)
        if len(img.shape) < 3:
            img = np.reshape(img, (448,448,1))
            img = np.repeat(img, (3), axis=-1)
        if img.shape[2] > 3:
            img = img[:,:,0:3]

        img = self.totensor(img)
        if self.is_training:
            img = self.randomerasing(img)
        img = self.norm(img)

        return img, label

    def __len__(self):
        return len(self.datalist)
