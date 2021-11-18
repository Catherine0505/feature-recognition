import torch
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage
from skimage.transform import resize
from scipy.ndimage import zoom
import scipy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

feature_points = []
images = []
file_names = os.listdir("imm_face_db")
file_names.sort()
count = 0


for name in file_names:
    if name[-3:] == "asf":
        feature = np.loadtxt("imm_face_db/" + name,
            skiprows = 16, max_rows = 58)[:, [2, 3]]
        feature_points.append(feature)
    if name[-3:] == "jpg":
        image = skio.imread("imm_face_db/" + name)
        images.append(image)


class TrainingDataset(Dataset):
    """Load nosetip detection training dataset."""

    def __init__(self, transform = None, display = True, rescale_factor = 1/8):
        """
        Args:
            transform: Potential transformations imposed on the image.
            display: Whether to display sampled training images.
            rescale factor: rescale the image to smaller size in order to
                decrease computation complexity.
        """
        if display:
            height, width = images[0].shape[:2]
            for i in range(4):
                plt.imshow(images[i])
                plt.plot(feature_points[i][-6, 0] * width,
                    feature_points[i][-6, 1] * height,
                    linestyle = "none", marker = ".")
                plt.show()

        self.feature_points = np.array(feature_points[:192])
        self.feature_points = self.feature_points[:, -6]
        self.feature_points = np.flip(self.feature_points, axis = 1).copy().astype(np.float32)

        self.images = np.array(images[:192])
        self.images = skimage.color.rgb2gray(self.images)
        self.images = zoom(self.images, [1, rescale_factor, rescale_factor], order = 1, mode = "reflect")
        self.images = self.images.astype(np.float32) - 0.5


    def __len__(self):
        return self.feature_points.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        feature = self.feature_points[idx]
        sample = {'image': image, 'feature': feature}

        return sample

class ValidationDataset(Dataset):
    """Load nosetip detection validation dataset."""

    def __init__(self, transform = None, rescale_factor = 1/8):
        """
        Args:
            transform: Potential transformations imposed on the image.
            rescale factor: rescale the image to smaller size in order to
                decrease computation complexity.
        """

        self.feature_points = np.array(feature_points[192:])
        self.feature_points = self.feature_points[:, -6]
        self.feature_points = np.flip(self.feature_points, axis = 1).copy().astype(np.float32)

        self.images = np.array(images[192:])
        self.images = skimage.color.rgb2gray(self.images)
        self.images = zoom(self.images, [1, rescale_factor, rescale_factor],
            order = 1, mode = "reflect")
        self.images = self.images.astype(np.float32) - 0.5


    def __len__(self):
        return self.feature_points.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        feature = self.feature_points[idx]
        sample = {'image': image, 'feature': feature}


        return sample

def training_dataloader(batch_size = 8, rescale_factor = 1/8):
    """
    Training dataloader. Returns the training dataset and torch.DataLoader
        object.
    """
    training_data = TrainingDataset(display =  False,
        rescale_factor = rescale_factor)
    return training_data, \
        DataLoader(training_data, batch_size = batch_size, shuffle=False)

def validation_dataloader(rescale_factor = 1/8):
    """
    Validation dataloader. Returns the validation dataset and torch.DataLoader
        object.
    """
    validation_data = ValidationDataset(rescale_factor = rescale_factor)
    return validation_data, \
        DataLoader(validation_data, batch_size = len(validation_data))
