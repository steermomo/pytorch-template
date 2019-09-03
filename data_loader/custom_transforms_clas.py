import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    GaussNoise,
    Cutout,
)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
        return {'image': img,
                'label': mask}


class RandomVertFlip:
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = cv2.rotate(img, rotate_degree)

        return {'image': img,
                'label': mask}


class RandomRoll:
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        r, c, ch = img.shape
        if random.random() < 0.5:
            sh = random.randint(-c, c)
            img = np.roll(img, sh, axis=1)

        return {'image': img,
                'label': mask}


class RandomGridTr:
    """ 需要放在最后执行"""

    def __init__(self, p=0.5):
        self.aug = GridDistortion(num_steps=10)
        # self.aug = ElasticTransform(p=p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        augmented = self.aug(image=img)
        image_elastic = augmented['image']
        # mask_elastic = augmented['mask']
        return {
            'image': image_elastic,
            'label': mask
        }


class RandomComp:
    """ 需要放在最后执行"""

    def __init__(self, p=0.5):
        self.aug = Compose([
            RandomBrightnessContrast(),
            RandomGamma(),
            CLAHE(),
            OneOf([
                GaussNoise(),
                Cutout(num_holes=10, max_h_size=10, max_w_size=10)
            ])
        ])

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        augmented = self.aug(image=img)
        image_elastic = augmented['image']
        # mask_elastic = augmented['mask']
        return {
            'image': image_elastic,
            'label': mask
        }


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size[0] * 0.5), int(self.base_size[0] * 1.1))
        h, w, _ = img.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # pad crop
        if short_size < self.crop_size[0]:
            padh = self.crop_size[0] - oh if oh < self.crop_size[0] else 0
            padw = self.crop_size[1] - ow if ow < self.crop_size[1] else 0
            img = np.pad(img, pad_width=(0, 0, padh, padw), constant_values=0)

        # random crop crop_size
        h, w, _ = img.shape
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])
        img = img[y1:y1 + self.crop_size[0], x1:x1 + self.crop_size[1]].copy()

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        h, w, _ = img.shape
        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]
            oh = int(1.0 * h * ow / w)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # center crop
        h, w, _ = img.shape
        x1 = int(round((w - self.crop_size[1]) / 2.))
        y1 = int(round((h - self.crop_size[0]) / 2.))
        img = img[y1:y1 + self.crop_size[0], x1:x1 + self.crop_size[1]].copy()

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'label': mask}
