import glob
import torch

from numpy.fft import fftn, fftshift
from skimage import io

class DeStripe_Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_3D = io.imread(path)
        self.mask_dir = path.split('/')[-1][:-4]
        self.cut = len(str(self.img_3D.shape[0]))

    def __len__(self):
        return self.img_3D.shape[0]

    def __getitem__(self, item):
        img = self.img_3D[item]

        img_name = str(item).zfill(self.cut)
        mask_path = f'./preprocessing/{self.mask_dir}/{img_name}.png'

        (h, w) = img.shape
        w_h = max(h, w)
        fft_img = fftshift(fftn(img, s=[w_h, w_h])/(w_h*w_h))

        cor_mask = io.imread(mask_path)

        # ndarray to tensor
        fft_img = torch.from_numpy(fft_img)
        cor_mask = torch.from_numpy(cor_mask/255)
        img_3D = torch.from_numpy(self.img_3D.astype('int16'))

        return img.astype('int16'), fft_img, cor_mask, h, w, w_h, img_3D
