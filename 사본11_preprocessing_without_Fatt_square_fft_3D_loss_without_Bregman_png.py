import argparse
import os
import numpy as np
import warnings

from numpy.fft import fftn, fftshift
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt


def preprocessing(opt):
    print('Generating corruption mask...')
    # load input 3D image (=2D images are stacked along z-axis)
    img_3D = io.imread(opt.input_path)

    (h, w) = img_3D.shape[1:]
    w_h = max(h, w)
    if opt.square_fft == True:
        output = np.zeros((img_3D.shape[0], w_h, w_h))
    else:
        output = np.zeros((img_3D.shape[0], h, w))

    img_name = opt.input_path.split('/')[-1][:-4]

    for i in range(img_3D.shape[0]):
        print(f'{i+1} / {img_3D.shape[0]}')
        img = img_3D[i]

        # To normalize, divided by h*w
        if opt.square_fft == True:
            fft_img = fftshift(fftn(img, s=[w_h, w_h]) / (w_h * w_h))
        else:
            fft_img = fftshift(fftn(img)/(h*w))

        #####################################################
        # # JUST for visualization of image in Fourier domain
        # FFT_image_im = np.log(np.abs(fft_img))  # magnitude
        # # Normalization: range 0~255
        # FFT_image_im = (FFT_image_im - np.min(FFT_image_im)) / (np.max(FFT_image_im) - np.min(FFT_image_im)) * 255
        # FFT_image_im = np.clip(np.round(FFT_image_im), 0, 255).astype(np.uint8)
        # plt.imshow(FFT_image_im, cmap='gray')
        # plt.show()
        # plt.imsave(f'{opt.temp_output_path}/{img_name}_fft.png', FFT_image_im, cmap='gray')
        #####################################################

        if opt.square_fft == True:
            x = np.arange(-w_h / 2, w_h / 2, dtype=np.float64)
            y = np.arange(-w_h / 2, w_h / 2, dtype=np.float64)
        else:
            x = np.arange(-h / 2, h / 2, dtype=np.float64)
            y = np.arange(-w / 2, w / 2, dtype=np.float64)

        [x, y] = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt((np.square(x) + np.square(y)))
        phi = np.arctan2(x, y)

        ###########################
        # # Display 'r'
        # r = np.clip(np.round((r - np.min(r))/(np.max(r) - np.min(r)) * 255), 0, 255)
        # plt.imshow(r)
        # plt.show()
        ###########################

        # Corruption Matrix
        Cor_Matrix = np.zeros(r.shape)
        len_r = int(np.max(r)) + 1
        for rr in tqdm(range(len_r)):
            # Magnitude
            fft_img_mag = np.abs(fft_img)

            # Calculate std of every annulus--> use them for whitening.
            annulus_list = fft_img_mag[np.where((r >= rr) & (r < rr+1))]
            std = np.std(annulus_list)

            '''
            Whitening means to make the variance of the distribution as 1.
            It doesn't contain making the mean of the distribution as 0.
            '''
            # Whitening and Rayleigh function: probability of being uncorrupted
            Cor_Matrix = np.where((r >= rr) & (r < rr+1), np.exp(-(fft_img_mag / std)**2/2), Cor_Matrix)

        ########################
        # # Display 'Corruption Matrix'
        # annulus = np.clip(np.round(Cor_Matrix*255), 0, 255)
        # plt.imshow(annulus, cmap='gray')
        # plt.show()
        # plt.imsave(f'{opt.temp_output_path}/{img_name}_matrix.png', annulus, cmap='gray')
        ########################

        ########################
        # # Display 'histogram of Corruption Matrix'
        # # To find a threshold value of corruption mask
        # plt.hist(Cor_Matrix.ravel(), bins=100)
        # plt.show()
        #######################

        # Corruption Mask
        Cor_Mask = np.where(Cor_Matrix > 0.001, 0, 1)
        Cor_Mask = np.clip(np.round(Cor_Mask * 255), 0, 255)

        #######################
        # # Display Corruption Mask
        # plt.imshow(Cor_Mask, cmap='gray')
        # plt.show()
        #######################

        # Save corruption mask
        output[i] = Cor_Mask

    io.imsave(f'{opt.output_path}/{img_name}_square_fft.png', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./dataset/DRIVE_raw_211002_SCAPE_casperGFP_4dpf_S1_R2_801planex1vol.tif')
    parser.add_argument('--temp_output_path', type=str, default='./temp')
    parser.add_argument('--output_path', type=str, default='./preprocessing')
    parser.add_argument('--square_fft', type=bool, default=True, help="if True, FFT mask is generated in square shape.")
    opt = parser.parse_args()

    if not os.path.exists(opt.output_path): os.makedirs(opt.output_path)
    if not os.path.exists(opt.temp_output_path): os.makedirs(opt.temp_output_path)

    warnings.filterwarnings(action='ignore')
    preprocessing(opt)
