import argparse
import os
import torch
import random
import torch.nn as nn
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt

from tqdm import tqdm
from 사본8_model_without_Fatt_square_fft_3D_loss import *
from 사본8_dataloader_without_Fatt_square_fft_3D_loss import DeStripe_Dataset
from external_code_utils_torch import *
from torch.utils.data import DataLoader
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam: learning rate')
parser.add_argument('--in_dir', type=str, default='./dataset/2frame_square_fft.tif', help="input directory")
parser.add_argument('--out_dir', type=str, default='./사본9_result')

parser.add_argument('--ring_width', type=int, default=2, help='width of each annulus ring (unit: pixel)')
parser.add_argument('--max_neigh', type=int, default=128, help='the number of cut-off neighbors if there are too many neighbors')
parser.add_argument('--beta', type=float, default=0.01, help='weight of the second loss term')
opt = parser.parse_args()

if not os.path.exists(opt.out_dir): os.makedirs(opt.out_dir)

warnings.filterwarnings(action='ignore')

# use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
train_data = DeStripe_Dataset(opt.in_dir)
train_loader = DataLoader(dataset=train_data, shuffle=False)

#################### Define parameters ####################
# Define model parameters
_, _, _, h, w, w_h, img_3D = next(iter(train_data))
in_features = 1
out_features = 16

# Define split Bregman parameters
mu = 0.1                # Lagrange multiplier
continuity_x = 1        # Pentalty parameters of continuity along x-axis
continuity_y = 1        # Pentalty parameters of continuity along y-axis
continuity_z = 1        # Pentalty parameters of continuity along z-axis
alpha = 1               # Trade-off parameter
############################################################

### Generate annulus index list in advance ------------------
# Create coordinate grid in polar
x = np.arange(-w_h / 2, w_h / 2, dtype=np.float64)
y = np.arange(-w_h / 2, w_h / 2, dtype=np.float64)
[x, y] = np.meshgrid(x, y, indexing='ij')

# Annuls ring
r = np.square(x) + np.square(y)
len_r = int(np.sqrt(np.square(w_h/2) + np.square(w_h/2))) + 1

# iter_r = total number of annulus rings
if len_r % opt.ring_width != 0:
    iter_r = len_r // opt.ring_width + 1
else:
    iter_r = len_r // opt.ring_width

# index_list: annulus index list prepared in advance
index_list = []
print("Start generating annulus index list...")
for rr in range(iter_r):
    # index: annulus region index
    index = np.where((r >= (rr * opt.ring_width)**2) & (r < ((rr + 1) * opt.ring_width)**2))
    index_list.append(index)
print("Done!")
# -----------------------------------------------------------

# Define model
model = GNN(in_features=in_features,
            out_features=out_features,
            max_neigh=opt.max_neigh,
            index_list=index_list,
            w_h=w_h)

# # If you use MULTIPLE GPUs, use 'DataParallel'
# model = nn.DataParallel(model)

# to device
model = model.to(device)

# loss function
criterion = nn.MSELoss().to(device)

# optimizer and scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

total_train_loss = []
first_train_loss = []
second_train_loss = []
epoch_graph = []

### Initialize for split Bregman ----------------
# B_xx = torch.zeros_like(img_3D, dtype=torch.float32)
# B_yy = B_xx
# B_zz = B_xx
# B_xy = B_xx
# B_yz = B_xx
# B_zx = B_xx
#
# Z_xx = xx(img_3D)
# Z_yy = yy(img_3D)
# Z_zz = zz(img_3D)
# Z_xy = xy(img_3D)
# Z_yz = yz(img_3D)
# Z_zx = zx(img_3D)
### ---------------------------------------------

img_3D = img_3D.type(torch.FloatTensor).to(device)
for e in range(opt.epoch):
    model.train()
    output_3D = torch.zeros((len(train_loader), h, w)).to(device)

    epoch_graph.append(e+1)
    second_loss = 0.0
    for idx, data in enumerate(tqdm(train_loader)):
        # fft_img, cor_mask, height=nx, width=ny, w_h
        img, fft_img, mask, _, _, _, _ = data

        # 3D to 2D (remove batch)
        fft_img = fft_img.squeeze()
        mask = mask.squeeze()
        output = torch.zeros(w_h, w_h, out_features, requires_grad=True).type(torch.complex128)

        fft_img = fft_img.to(device)
        mask = mask.to(device)
        output = output.to(device)
        output = model(input_image=fft_img, input_mask=mask, output=output)

        output = output[:h, :w].type(torch.cuda.FloatTensor)

        # to calculate the first loss term (L1 loss)
        output_3D[idx] = output

        # to calculate the second loss term
        fft_output = torch.fft.fftshift(torch.fft.fftn(output, s=[w_h, w_h])/(w_h*w_h))
        for index in tqdm(index_list):
            annulus_vector = fft_output[index]
            uncor_neigh = annulus_vector[torch.where(mask[index] == 0)]
            cor_neigh = annulus_vector[torch.where(mask[index] == 1)]

            # corrupted neighbor의 수가 0개일 때 mean을 계산하면 nan이 되므로 예외처리
            if len(cor_neigh) == 0:
                cor_neigh_mean = 0
            else:
                cor_neigh_mean = torch.abs(cor_neigh).mean()
            for un in uncor_neigh:
                second_loss_term = (torch.abs(un) - cor_neigh_mean)**2
                second_loss += second_loss_term

    ### Split Bregman ------------------------
    # X = output_3D.detach().cpu().numpy()
    # X_xx = xx(X)
    # X_yy = yy(X)
    # X_zz = zz(X)
    # X_xy = xy(X)
    # X_yz = yz(X)
    # X_zx = zx(X)
    #
    # # Update Z
    # Z_xx = shrink(continuity_x * X_xx + B_xx, alpha / mu)
    # Z_yy = shrink(continuity_y * X_yy + B_yy, alpha / mu)
    # Z_zz = shrink(continuity_z * X_zz + B_zz, alpha / mu)
    # Z_xy = shrink(2 * math.sqrt(continuity_x * continuity_y) * X_xy + B_xy, alpha / mu)
    # Z_yz = shrink(2 * math.sqrt(continuity_y * continuity_z) * X_yz + B_yz, alpha / mu)
    # Z_zx = shrink(2 * math.sqrt(continuity_z * continuity_x) * X_zx + B_zx, alpha / mu)
    #
    # # Update B
    # B_xx = B_xx + X_xx - Z_xx
    # B_yy = B_yy + X_yy - Z_yy
    # B_zz = B_zz + X_zz - Z_zz
    # B_xy = B_xy + X_xy - Z_xy
    # B_yz = B_yz + X_yz - Z_yz
    # B_zx = B_zx + X_zx - Z_zx
    ### --------------------------------------

    optimizer.zero_grad()

    loss = criterion(output_3D, img_3D)
    # loss1 = criterion(Z_xx, continuity_x*X_xx + B_xx)
    # loss2 = criterion(Z_yy, continuity_y*X_yy + B_yy)
    # loss3 = criterion(Z_zz, continuity_z * X_zz + B_zz)
    # loss4 = criterion(Z_xy, 2 * math.sqrt(continuity_x * continuity_y) * X_xy + B_xy)
    # loss5 = criterion(Z_yz, 2 * math.sqrt(continuity_y * continuity_z) * X_yz + B_yz)
    # loss6 = criterion(Z_zx, 2 * math.sqrt(continuity_z * continuity_x) * X_zx + B_zx)
    total_loss = loss + opt.beta * second_loss
    total_loss_item = total_loss.item()
    first_loss_item = loss.item()
    second_loss_item = second_loss.item()

    total_loss.backward()

    optimizer.step()

    print(f'Epoch: {e+1} / {opt.epoch}, Total loss: {total_loss_item:.3f}, First loss: {first_loss_item:.3f}, Second loss: {second_loss_item:.3f}')
    total_train_loss.append(total_loss_item)
    first_train_loss.append(first_loss_item)
    second_train_loss.append(second_loss_item)

    # Save the plot and output image #########################
    plt.figure()
    plt.plot(np.array(epoch_graph), np.array(total_train_loss), label='total')
    plt.plot(np.array(epoch_graph), np.array(first_train_loss), label='loss1')
    plt.plot(np.array(epoch_graph), np.array(second_train_loss), label='loss2')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'./{opt.out_dir}/train_loss_curve.png')

    # plt.figure()
    # plt.plot(np.array(epoch_graph), np.array(second_train_loss))
    # plt.xlabel('Epoch')
    # plt.ylabel('Second loss')
    # plt.savefig(f'./{opt.out_dir}/train_second_loss_curve.png')

    output_3D_save = output_3D.detach().cpu().numpy()
    io.imsave(f'./{opt.out_dir}/{e}.tif', output_3D_save)
    #########################################################

    if (e+1) % 20 == 0:
        torch.save(model.state_dict(), f'{opt.out_dir}/model_{e+1}_{total_loss_item:.3f}.pth')

# # batch mode algorithm
# for e in range(epoch):
#     for batch in dataloader:
#         out_img = torch.ones_like(batch)
#         for data in batch:
#             for r in circle:
#                 out = GNN
#                 out_img[?] = out
#         loss.backward()
