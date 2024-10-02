"""
Data Augmentations
"""

import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.transforms import RandAffined, RandAxisFlipd, Compose, RandFlip, RandZoom, RandGaussianNoise, RandAdjustContrast, RandGaussianSharpen


# credit CKD-TransBTS
class DataAugmenter(nn.Module):
    def __init__(self):
        super(DataAugmenter, self).__init__()   
        self.flip_dim = []
        self.zoom_rate = uniform(0.7, 1.0)
        self.sigma_1 = uniform(0.5, 1.5)
        self.sigma_2 = uniform(0.5, 1.5)
        self.image_zoom = Zoom(zoom=self.zoom_rate, mode="trilinear", padding_mode="constant")
        self.label_zoom = Zoom(zoom=self.zoom_rate, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=uniform(0, 0.33))
        self.blur = GaussianSharpen(sigma1=self.sigma_1, sigma2=self.sigma_2)
        self.contrast = AdjustContrast(gamma=uniform(0.65, 1.5))
    
    def forward(self, images, lables):
        with torch.no_grad():
            for b in range(images.shape[0]):
                image = images[b].squeeze(0)
                lable = lables[b].squeeze(0)
                if random() < 0.15:
                    image = self.image_zoom(image)
                    lable = self.label_zoom(lable)
                if random() < 0.5:
                    image = torch.flip(image, dims=(1,))
                    lable = torch.flip(lable, dims=(1,))
                if random() < 0.5:
                    image = torch.flip(image, dims=(2,))
                    lable = torch.flip(lable, dims=(2,))
                if random() < 0.5:
                    image = torch.flip(image, dims=(3,))
                    lable = torch.flip(lable, dims=(3,))
                if random() < 0.15:
                    image = self.noisy(image)
                if random() < 0.15:
                    image = self.blur(image)
                if random() < 0.15:
                    image = self.contrast(image)
                images[b] = image.unsqueeze(0)
                lables[b] = lable.unsqueeze(0)
            return images, lables

class DataAugmenter_tmp(nn.Module):
    def __init__(self):
        super(DataAugmenter_tmp, self).__init__()

        # Define the augmentation pipeline using Compose
        self.augmentations = Compose([
            RandZoom(min_zoom=0.7, max_zoom=1.0, prob=0.15, mode=("trilinear", "nearest")),
            RandFlip(spatial_axis=1, prob=0.5),
            RandFlip(spatial_axis=2, prob=0.5),
            RandFlip(spatial_axis=3, prob=0.5),
            RandGaussianNoise(prob=0.15, mean=0, std=0.33),
            RandGaussianSharpen(sigma1_x=(0.5, 1.5), sigma2_x=(0.5, 1.5), prob=0.15),
            RandAdjustContrast(gamma=(0.65, 1.5), prob=0.15),
        ])

    def forward(self, images, labels):
        with torch.no_grad():
            for b in range(images.shape[0]):
                # Apply augmentations on both image and label
                augmented = self.augmentations({"image": images[b], "label": labels[b]})
                images[b], labels[b] = augmented["image"], augmented["label"]

        return images, labels

class AttnUnetAugmentation(nn.Module):
    def __init__(self):
      super(AttnUnetAugmentation, self).__init__()
      self.axial_prob = uniform(0.1, 0.6)
      self.affine_prob = uniform(0.1, 0.5)
      self.crop_prob = uniform(0.1, 0.5)
      self.axial_flips = RandAxisFlipd(keys=["image", "label"], prob=self.axial_prob)
      self.affine = RandAffined(
          keys=["image", "label"],
          mode=("bilinear", "nearest"),
          prob=self.affine_prob,
          shear_range=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1),
          padding_mode="border",
      )

    def forward(self, data):
      with torch.no_grad():
        data = self.affine(data)
        data = self.axial_flips(data)
        return data