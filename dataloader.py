import os
import cv2
import torch

from numpy.fft import fftn, fftshift
from skimage import io

class DeStripe_Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_3D = io.imread(path)

        image_name = path.split('/')[-1]
        self.cor_mask = io.imread(f'./preprocessing/{image_name}')

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]
        (h, w) = img.shape
        fft_img = fftshift(fftn(img)/(h*w))

        cor_mask = self.cor_mask[item]

        # ndarray to tensor
        fft_img = torch.from_numpy(fft_img)
        cor_mask = torch.from_numpy(cor_mask/255)

        return self.img_3D.astype('int16'), fft_img, cor_mask, h, w
