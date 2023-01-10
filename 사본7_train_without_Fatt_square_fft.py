import argparse
import os
import torch
import random
import torch.nn as nn
import numpy as np
import warnings
import matplotlib.pyplot as plt

from tqdm import tqdm
from 사본7_model_without_Fatt_square_fft import *
from 사본7_dataloader_without_Fatt_square_fft import DeStripe_Dataset
from torch.utils.data import DataLoader
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam: learning rate')
parser.add_argument('--in_dir', type=str, default='./dataset/2frame_square_fft.tif', help="input directory")
parser.add_argument('--out_dir', type=str, default='./사본7_result_MSEloss')

parser.add_argument('--ring_width', type=int, default=2, help='width of each annulus ring (unit: pixel)')
parser.add_argument('--max_neigh', type=int, default=128, help='the number of cut-off neighbors if there are too many neighbors')
opt = parser.parse_args()

if not os.path.exists(opt.out_dir): os.makedirs(opt.out_dir)

warnings.filterwarnings(action='ignore')

# use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader
train_data = DeStripe_Dataset(opt.in_dir)
train_loader = DataLoader(dataset=train_data, shuffle=True)

# Define model
_, _, _, h, w, w_h = next(iter(train_data))
in_features = 1
out_features = 16
model = GNN(in_features=in_features,
            out_features=out_features,
            ring_width=opt.ring_width,
            max_neigh=opt.max_neigh,
            w_h=w_h)

# # If you use MULTIPLE GPUs, use 'DataParallel'
# model = nn.DataParallel(model)

# to device
model = model.to(device)

# loss function
criterion = nn.L1Loss().to(device)

# optimizer and scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_loss = []
epoch_graph = []
output_3D = np.zeros((len(train_loader), h, w))

for e in range(opt.epoch):
    model.train()

    total_train_loss = 0.0
    epoch_graph.append(e+1)
    for idx, data in enumerate(tqdm(train_loader)):
        # fft_img, cor_mask, height=nx, width=ny, w_h
        img, fft_img, mask, _, _, _ = data
        img = img.to(device)

        # 3D to 2D (remove batch)
        fft_img = fft_img.squeeze()
        mask = mask.squeeze()
        output = torch.zeros(w_h, w_h, out_features, requires_grad=True).type(torch.complex128)

        fft_img = fft_img.to(device)
        mask = mask.to(device)
        output = output.to(device)
        output = model(input_image=fft_img, input_mask=mask, output=output)

        optimizer.zero_grad()

        output = output[:h, :w]
        loss = criterion(output, img)
        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        output_3D[idx] = output.detach().cpu().numpy()

    print(f'Epoch: {e+1} / {opt.epoch}, Loss: {total_train_loss:.3f}')
    train_loss.append(total_train_loss)

    plt.figure()
    plt.plot(np.array(epoch_graph), np.array(train_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'./{opt.out_dir}/train_loss_curve.png')

    io.imsave(f'./{opt.out_dir}/{e}.tif', output_3D)

# # batch mode algorithm
# for e in range(epoch):
#     for batch in dataloader:
#         out_img = torch.ones_like(batch)
#         for data in batch:
#             for r in circle:
#                 out = GNN
#                 out_img[?] = out
#         loss.backward()
