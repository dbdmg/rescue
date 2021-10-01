import math
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from skimage import transform
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, dilation

class ToTensor(object):
    """
    Transform object used to swap the axes of images and masks from normal convention to PyTorch convention.
    The image and mask are received as dict['image'] and dict['mask']
    Swap from:
        images: H x W x C -> C x H x W
        masks: no swapping
    """
    def __init__(self, round_mask: bool, to_float=True):
        """
        Args:
            round_mask (bool): if True, apply a np.rint() to the mask to round it to the nearest integer
            to_float (bool): if True, cast tensor to float.
        """
        self.round_mask = round_mask
        self.to_float = to_float
        return

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.round_mask:
            mask = np.rint(mask)

        result = {}

        #swap axis
        #numpy: H x W x C
        #torch: C x H x W
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        result['image'] = torch.from_numpy(image)
        result['mask'] = torch.from_numpy(mask)

        if self.to_float:
            result['image'] = result['image'].float()
            result['mask'] = result['mask'].float()

        return result

class Normalize(object):
    def __init__(self, t1, t2):
        self.norm = transforms.Normalize(t1, t2)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        result = {}

        image = self.norm(image)
        result['image'] = image
        result['mask'] = mask
        return result

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        result = {}
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        image = F.interpolate(image, size=self.size).squeeze(0)
        mask = F.interpolate(mask, size=self.size).squeeze(0)
        result['image'] = image
        result['mask'] = mask
        return result
    
class NormalizePerImage(object):
    def __init__(self, mean: bool, std: bool, exclude_channels):
        self.mean = mean
        self.std = std
        self.exclude_channels = exclude_channels
        return

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        result = {}
        n_channels = image.shape[0]
        dim_list = tuple(x for x in range(1, len(image.shape)))

        if self.mean:
            mean_per_channel = image.mean(dim=dim_list, keepdim=True)
            if self.exclude_channels is not None:
                if isinstance(self.exclude_channels, bool) and self.exclude_channels:
                    mean_per_channel[-1, :] = 0
                elif isinstance(self.exclude_channels, list):
                    mean_per_channel[self.exclude_channels, :] = 0
            image = image - mean_per_channel

        if self.std:
            std_per_channel = image.std(dim=dim_list, keepdim=True)
            if self.exclude_channelsis is not None:
                if isinstance(self.exclude_channels, bool) and self.exclude_channels:
                    std_per_channel[-1, :] = 1
                elif isinstance(self.exclude_channels, list):
                    std_per_channel[self.exclude_channels, :] = 1

            image = image / std_per_channel

        result['image'] = image
        result['mask'] = mask
        return result


class Rescale(object):
    """
    Transform object used to resize the images and masks, eventually changing its aspect ratio.
    The image and mask are received as dict['image'] and dict['mask'].
    """
    def __init__(self, output_size):
        """
        Constructor of Rescale object.

        Args:
            output_size(int/tuple(int, int)): if int, a square image is returned. Otherwise, the image is rescaled. Tuple is in (h, w) format.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']

        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), new_w

        img = transform.resize(img, (new_h, new_w), preserve_range=True).astype('float32')
        mask = transform.resize(mask, (new_h, new_w), preserve_range=True).astype('float32')

        return {'image': img, 'mask': mask}

class RandomCrop(object):
    """
    Transform object to perform the same random crop to both image and mask.
    The image and mask are received as dict['image'] and dict['mask'].
    """
    def __init__(self, probability, output_size, seed=None):
        """
        Args:
            probability(float): probability to perform the random crop
            output_size(int/tuple(int, int)): dimensions to perform the crop. If tuple, the dimensions are (h, w)
        """
        assert isinstance(probability, (float, int))
        assert isinstance(output_size, (int, tuple))
        assert probability >= 0 and probability <= 1

        self.probability = probability
        self.output_size = output_size
        seed2 = seed + hash(self) if seed is not None else None
        self.r = random.Random(seed2)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']

        r = self.r.random()
        if r > self.probability:
            return sample
        
        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = self.r.randint(0, h - new_h)
        left = self.r.randint(0, w - new_w)

        img = img[top:top + new_h, left:left + new_w]
        mask = mask[top:top + new_h, left:left + new_w]

        return {'image': img, 'mask': mask}

    def __hash__(self):
        return 1

class RandomRotate(object):
    """
    Randomly rotate both the image and mask.
    The image and mask are received as dict['image'] and dict['mask']
    """
    def __init__(self, probability, rotation, seed=None):
        """
        Args:
            probability(float): probability to perform the random rotation
            rotation(int/tuple(int, int)): rotation angle. 
                                            If int, rotation is randomly picked from -rotation, +rotation. 
                                            If tuple, (min, max) notation is used.
        """
        assert isinstance(probability, (float, int))
        assert probability >= 0 and probability <= 1
        assert isinstance(rotation, (int, tuple))
        self.probability = probability
        self.rotation = rotation
        seed2 = seed + hash(self) if seed is not None else None
        self.r = random.Random(seed2)

        if isinstance(rotation, int):
            self.min_rotation = -rotation
            self.max_rotation = rotation
        elif isinstance(rotation, tuple):
            self.min_rotation = rotation[0]
            self.max_rotation = rotation[1]

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        result = {}

        if self.r.random() < self.probability:
            angle = self.r.randint(self.min_rotation, self.max_rotation)
            img = transform.rotate(img, angle, preserve_range=True, mode='reflect').astype('float32') #TF.rotate(img, angle)
            mask = transform.rotate(mask, angle, preserve_range=True, mode='reflect').astype('float32') #TF.rotate(mask, angle)

            result['image'] = img
            result['mask'] = mask

        result['image'] = img
        result['mask'] = mask

        return result

    def __hash__(self):
        return 2

class RandomHorizontalFlip(object):
    """
    Randomly perform horizontal flip on both image and mask.
    The image and mask are received as dict['image'] and dict['mask'].
    """
    def __init__(self, probability, seed=None):
        """
        Args:
            probability(float): probability to perform the random horizontal flipping
        """
        assert isinstance(probability, (float, int))
        assert probability >= 0 and probability <= 1

        self.probability = probability
        seed2 = seed + hash(self) if seed is not None else None
        self.r = random.Random(seed2)
        return

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']

        result = {}

        if self.r.random() < self.probability:
            img = np.fliplr(img).copy() #TF.hflip(img)
            mask = np.fliplr(mask).copy() #TF.hflip(mask)

        result['image'] = img
        result['mask'] = mask

        return result

    def __hash__(self):
        return 3

class RandomVerticalFlip(object):
    """
    Randomly flips vertically both the image and mask.
    The image and mask are received as dict['image'] and dict['mask'].
    """
    def __init__(self, probability, seed=None):
        """
        Args:
            probability(float): probability to randomly flip both image and mask vertically.
        """
        assert isinstance(probability, (float, int))
        assert probability >= 0 and probability <= 1

        self.probability = probability
        seed2 = seed + hash(self) if seed is not None else None
        self.r = random.Random(seed2)

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        result = {}

        if self.r.random() < self.probability:
            img = np.flipud(img).copy() #TF.vflip(img)
            mask = np.flipud(mask).copy()  #TF.vflip(mask)

            result['image'] = img
            result['mask'] = mask

        result['image'] = img
        result['mask'] = mask

        return result

    def __hash__(self):
        return 4

class RandomShear(object):
    """
    Randomly apply shear to both image and mask.
    The image and mask are received as dict['image'] and dict['mask'].
    """
    def __init__(self, probability, angle, seed=None):
        """
        Args:
            probability(float): probability to apply the random shear transformation
            angle(float): angle expressed in degrees to apply the shear transformation
        """
        assert isinstance(probability, (float, int))
        assert isinstance(angle, (int, float))
        assert probability >= 0 and probability <= 1        

        self.probability = probability
        seed2 = seed + hash(self) if seed is not None else None
        self.r = random.Random(seed2)

        if isinstance(angle, int):
            self.angle = math.radians(angle)
            self.min_rotation = -self.angle
            self.max_rotation = self.angle
        elif isinstance(angle, tuple):
            self.angle = angle
            self.min_rotation = math.radians(angle[0])
            self.max_rotation = math.radians(angle[1])

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        result = {}

        if self.r.random() < self.probability:
            angle = self.r.random() * (self.max_rotation - self.min_rotation) + self.min_rotation
            tr = transform.AffineTransform(shear=angle)
            img = transform.warp(img, tr, mode='reflect', preserve_range=True).astype('float32')
            mask = transform.warp(mask, tr, mode='reflect', preserve_range=True).astype('float32')

            result['image'] = img
            result['mask'] = mask

        result['image'] = img
        result['mask'] = mask

        return result

    def __hash__(self):
        return 5