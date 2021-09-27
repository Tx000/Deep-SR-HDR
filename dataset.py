from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
import torch
from prefetch_generator import BackgroundGenerator
import h5py


class HDR_Dataset(Dataset):
    def __init__(self, datadir, crop_size, scale_factor=2):
        hf = h5py.File(datadir)
        self.inputs = hf.get('IN')
        self.labels = hf.get('GT')
        self.num = self.inputs.shape[0]
        self.h = self.inputs.shape[2]
        self.w = self.inputs.shape[3]
        self.scale_factor = scale_factor
        self.crop_size = crop_size

    def __getitem__(self, index):
        in_imgs = self.inputs[index, :, :, :]
        ref_HDR = self.labels[index, :, :, :]
        in_imgs, ref_HDR = self.random_crop(in_imgs, ref_HDR, crop_size=self.crop_size)
        in_imgs, ref_LR_HDR, ref_HR_HDR = self.augment(in_imgs, ref_HDR, scale_factor=self.scale_factor)
        return torch.from_numpy(in_imgs), torch.from_numpy(ref_LR_HDR), torch.from_numpy(ref_HR_HDR)

    def __len__(self):
        return self.num

    def random_crop(self, in_imgs, ref_HDR, crop_size):
        crop_h_start = random.randint(0, self.h - crop_size[0] - 1)
        crop_w_start = random.randint(0, self.w - crop_size[1] - 1)
        in_imgs = in_imgs[:, crop_h_start: crop_h_start + crop_size[0], crop_w_start: crop_w_start + crop_size[1]]
        ref_HDR = ref_HDR[:, crop_h_start: crop_h_start + crop_size[0], crop_w_start: crop_w_start + crop_size[1]]
        return in_imgs, ref_HDR

    def augment(self, in_imgs, ref_HDR, scale_factor=2):
        in_imgs = np.transpose(in_imgs, axes=(1, 2, 0))
        ref_HR_HDR = np.transpose(ref_HDR, axes=(1, 2, 0))
        in_imgs = np.clip(cv2.resize(in_imgs, dsize=(self.crop_size[1] // scale_factor, self.crop_size[0] // scale_factor),
                                     interpolation=cv2.INTER_CUBIC), 0, 1)

        flip = random.randint(0, 1)
        if flip == 1:
            in_imgs = cv2.flip(in_imgs, 1)
            ref_HR_HDR = cv2.flip(ref_HR_HDR, 1)

        rotate = random.randint(0, 3)
        if rotate != 0:
            mat = cv2.getRotationMatrix2D((self.crop_size[1] // scale_factor // 2, self.crop_size[0] // scale_factor // 2), 90*rotate, 1)
            in_imgs = cv2.warpAffine(in_imgs, mat, (self.crop_size[1] // scale_factor, self.crop_size[0] // scale_factor))

            mat = cv2.getRotationMatrix2D((self.crop_size[1] // 2, self.crop_size[0] // 2), 90 * rotate, 1)
            ref_HR_HDR = cv2.warpAffine(ref_HR_HDR, mat, (self.crop_size[1], self.crop_size[0]))

        ref_LR_HDR = np.clip(cv2.resize(ref_HR_HDR, dsize=(self.crop_size[1] // scale_factor, self.crop_size[0] // scale_factor),
                                        interpolation=cv2.INTER_LINEAR), 0, 1)

        in_imgs = np.transpose(in_imgs, axes=(2, 0, 1))
        ref_LR_HDR = np.transpose(ref_LR_HDR, axes=(2, 0, 1))
        ref_HR_HDR = np.transpose(ref_HR_HDR, axes=(2, 0, 1))
        return in_imgs, ref_LR_HDR, ref_HR_HDR


class HDR_DataLoader_pre(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
