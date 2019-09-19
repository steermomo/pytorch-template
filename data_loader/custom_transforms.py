import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import albumentations as alb
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


class ToTensorSdimMask(object):
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


class ToTensorMdimMask(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class Noop:
    def __call__(self, sample):
        return sample


class MaskToMultiDim:
    def __init__(self, nclass=5):
        self.nclass = nclass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        r, c, _ = img.shape
        target = np.zeros((r, c, self.nclass))

        for c_idx in range(self.nclass):
            target[:, :, c_idx] = np.where(mask == c_idx, 1, 0)

        return {
            'image': img,
            'label': target
        }


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomVertFlip:
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            # img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
        self.aug = alb.Rotate(limit=degree)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        augmented = self.aug(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        return {'image': img,
                'label': mask}


class RandomRoll:
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rolled = False
        r, c, ch = img.shape
        if random.random() < 0.5:
            sh = random.randint(-c, c)
            img = np.roll(img, sh, axis=1)
            mask = np.roll(mask, sh, axis=1)
            # rolled = True
        # if random.random() < 0.5 and not rolled:
        #     sh = random.randint(-r, r)
        #     img = np.roll(img, sh, axis=0)
        #     mask = np.roll(mask, sh, axis=0)
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
        # img = np.array(img).astype(np.uint8)
        # mask = np.array(mask).astype(np.uint8)

        augmented = self.aug(image=img, mask=mask)
        image_elastic = augmented['image']
        mask_elastic = augmented['mask']
        return {
            'image': image_elastic,
            'label': mask_elastic
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
                Cutout(num_holes=10, max_h_size=5, max_w_size=5)
            ])
        ])

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # img = np.array(img).astype(np.uint8)
        # mask = np.array(mask).astype(np.float32)

        augmented = self.aug(image=img, mask=mask)
        image_elastic = augmented['image']
        mask_elastic = augmented['mask']
        return {
            'image': image_elastic,
            'label': mask_elastic
        }


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.base_height = base_size[0]
        self.fill = fill
        self.aug = alb.Compose([
            alb.RandomScale(),
            alb.PadIfNeeded(min_height=base_size[0], min_width=base_size[1], border_mode=cv2.BORDER_REFLECT101),
            alb.RandomCrop(height=base_size[0], width=base_size[1])
        ])

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(f'===>{img.shape}')
        augmented = self.aug(image=img, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        return {
            'image': image_aug,
            'label': mask_aug
        }


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.aug = alb.Compose([
            alb.Resize(height=crop_size[0], width=crop_size[1])
        ])

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        augmented = self.aug(image=img, mask=mask)

        image_aug = augmented['image']
        mask_aug = augmented['mask']

        return {
            'image': image_aug,
            'label': mask_aug
        }


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
