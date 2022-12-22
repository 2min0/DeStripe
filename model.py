import random

import numpy as np
import torch
import torch.nn as nn
from torch.fft import ifftn, ifftshift


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return nn.functional.relu(x.real) + 1.j * nn.functional.relu(x.imag)


class Fatt(nn.Module):
    def __init__(self, out_features):
        super(Fatt, self).__init__()
        # L1 and Lw shape = (out_features+2, out_features)
        # '+2' means ro and theta (frequency information)
        self.L1_re = nn.Parameter(torch.randn(out_features + 2, out_features, requires_grad=True))
        self.L1_im = nn.Parameter(torch.randn(out_features + 2, out_features, requires_grad=True))
        self.L2_re = nn.Parameter(torch.randn(out_features + 2, out_features, requires_grad=True))
        self.L2_im = nn.Parameter(torch.randn(out_features + 2, out_features, requires_grad=True))

    def forward(self, h_p, ro_theta):
        h_p_re = h_p.real.type(torch.float32)
        h_p_im = h_p.imag.type(torch.float32)

        # h_p_comb = concatenate h_p and ro_theta (shape: (num_a, out_features))
        h_p_comb = torch.concat([h_p, ro_theta], dim=1)
        h_p_comb_re = h_p_comb.real.type(torch.float32)
        h_p_comb_im = h_p_comb.imag.type(torch.float32)

        # matrix multiplication: mul1 = h_p_comb @ L1
        mul1_re = torch.mm(h_p_comb_re, self.L1_re) - torch.mm(h_p_comb_im, self.L1_im)
        mul1_im = torch.mm(h_p_comb_re, self.L1_im) + torch.mm(h_p_comb_im, self.L1_re)

        # matrix multiplication: mul2 = h_p_comb @ L2
        mul2_re = torch.mm(h_p_comb_re, self.L2_re) - torch.mm(h_p_comb_im, self.L2_im)
        mul2_im = torch.mm(h_p_comb_re, self.L2_im) + torch.mm(h_p_comb_im, self.L2_re)

        # matrix multiplication: mul = mul1.T @ mul2
        # output shape: (out_features, out_features)
        mul_re = torch.mm(mul1_re.T, mul2_re) - torch.mm(mul1_im.T, mul2_im)
        mul_im = torch.mm(mul1_re.T, mul2_im) + torch.mm(mul1_im.T, mul2_re)

        # apply sigmoid
        mul_re = torch.sigmoid(mul_re)
        mul_im = torch.sigmoid(mul_im)

        # matrix multiplication: h_p @ mul
        out_re = torch.mm(h_p_re, mul_re) - torch.mm(h_p_im, mul_im)
        out_im = torch.mm(h_p_re, mul_im) + torch.mm(h_p_im, mul_re)
        out = out_re + 1.j*out_im

        return out


class FGNN(nn.Module):
    def __init__(self, meshgrid, in_features, out_features, ring_width, max_neigh, h, w):
        super(FGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w1_re = nn.Parameter(torch.randn(in_features, out_features, requires_grad=True))
        self.w2_re = nn.Parameter(torch.randn(in_features, out_features, requires_grad=True))
        self.w1_im = nn.Parameter(torch.randn(in_features, out_features, requires_grad=True))
        self.w2_im = nn.Parameter(torch.randn(in_features, out_features, requires_grad=True))

        # Create coordinate grid in polar
        self.meshgrid = meshgrid
        x = meshgrid[0]
        y = meshgrid[1]

        # Annuls ring
        self.r = torch.sqrt(torch.square(x) + torch.square(y))
        self.phi = torch.arctan2(x, y)
        self.ring_width = ring_width
        len_r = int(np.sqrt(np.square(h/2) + np.square(w/2))) + 1

        # iter_r = total number of annulus rings
        if len_r % ring_width != 0:
            self.iter_r = len_r // ring_width + 1
        else:
            self.iter_r = len_r // ring_width

        # the cut-off value of the number of neighbors
        self.max_neigh = max_neigh

        self.Fatt = Fatt(out_features)

    # Uncorrupted part
    def uncor_operator(self, h_p, neighbor):
        # complex number = {real} + j{imaginary}
        h_p_re = h_p.real.type(torch.float32)
        h_p_im = h_p.imag.type(torch.float32)
        neigh_re = neighbor.real.type(torch.float32)
        neigh_im = neighbor.imag.type(torch.float32)

        # complex: for a+bi and c+di, re = ac-bd, im = ad + bc
        mul_re = torch.mm(h_p_re, self.w1_re) - torch.mm(h_p_im, self.w1_im)
        mul_im = torch.mm(h_p_re, self.w1_im) + torch.mm(h_p_im, self.w1_re)
        mul = mul_re + 1.j*mul_im

        sum_re = torch.mm(neigh_re, self.w1_re) - torch.mm(neigh_im, self.w1_im)
        sum_re = (torch.sum(sum_re, dim=0) / neighbor.shape[0]).view(1,-1)
        sum_im = torch.mm(neigh_re, self.w1_im) + torch.mm(neigh_im, self.w1_re)
        sum_im = (torch.sum(sum_im, dim=0) / neighbor.shape[0]).view(1,-1)
        sum = sum_re + 1.j*sum_im

        out = 0.5 * (mul + sum)

        return out, sum

    # Corrupted part
    def cor_operator(self, h_p, sum):
        # complex number = {real} + j{imaginary}
        h_p_re = h_p.real.type(torch.float32)
        h_p_im = h_p.imag.type(torch.float32)

        # complex: for a+bi and c+di, re = ac-bd, im = ad + bc
        mul_re = torch.mm(h_p_re, self.w2_re) - torch.mm(h_p_im, self.w2_im)
        mul_im = torch.mm(h_p_re, self.w2_im) + torch.mm(h_p_im, self.w2_re)
        mul = mul_re + 1.j*mul_im

        out = mul - sum

        return out


    def forward(self, input_image, input_mask, output):
        num_a_list = []
        for rr in range(self.iter_r):
            # index: annulus region index, annulus_vector: annulus region in input image
            index = torch.where((self.r >= rr*self.ring_width) & (self.r < (rr+1)*self.ring_width))
            annulus_vector = input_image[index].view(-1, self.in_features)

            # uncor_neigh: uncorrupted region of annulus, cor_neigh: corrupted region of annulus
            # uncor_index: [1 1 1 0], cor_index: [0 0 0 1]
            # uncor_neigh: [u u u 0], cor_neigh: [0 0 0 c]
            uncor_index = torch.where(input_mask[index] == 0, 1, 0).view(-1, 1)
            uncor_index_a = uncor_index.expand(-1, self.in_features)
            uncor_neigh = annulus_vector * uncor_index_a

            cor_index = torch.where(input_mask[index] == 1, 1, 0).view(-1, 1)
            cor_index_a = cor_index.expand(-1, self.in_features)
            cor_neigh = annulus_vector * cor_index_a

            # for matrix operation, 1D --> 2D
            uncor_neigh = uncor_neigh.view(-1, self.in_features)
            cor_neigh = cor_neigh.view(-1, self.in_features)

            ################## prepare for Fatt #####################
            # the number of pixels in an annulus
            num_a = annulus_vector.cpu().shape[0]
            # ro, theta = frequency information
            ro = self.r[index]
            theta = self.phi[index]
            # ro_theta.shape = (num_a, 2)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            ro_theta = torch.stack([ro, theta], dim=-1).view(num_a, 2)
            ro_theta.to(device)
            #########################################################

            if uncor_neigh.shape[0] == 0:
                pass

            elif uncor_neigh.shape[0] > self.max_neigh:
                # If the number of neighbors is too large, randomly choose some of them.
                random_list = random.sample(range(0, len(uncor_neigh)-1), self.max_neigh)
                random_neigh = uncor_neigh[random_list].reshape(self.max_neigh, self.in_features)

                out, sum = self.uncor_operator(h_p=uncor_neigh, neighbor=random_neigh)
                uncor_index_f = uncor_index.view(-1, 1).float().expand(-1, self.out_features)
                uncor_out_re = out.real.type(torch.float32)
                uncor_out_im = out.imag.type(torch.float32)

                uncor_out = (uncor_index_f * uncor_out_re) + 1.j*(uncor_index_f * uncor_out_im)
                del uncor_index_f, uncor_out_re, uncor_out_im

                out = self.cor_operator(h_p=cor_neigh, sum=sum)
                cor_index_f = cor_index.view(-1, 1).float().expand(-1, self.out_features)
                cor_out_re = out.real.type(torch.float32)
                cor_out_im = out.imag.type(torch.float32)

                cor_out = (cor_index_f * cor_out_re) + 1.j * (cor_index_f * cor_out_im)
                del cor_index_f, cor_out_re, cor_out_im

                out = uncor_out + cor_out

                ########### Fatt ###########
                out = self.Fatt(out, ro_theta)
                ############################

                output[index] = out.type(torch.complex128)

            else:
                out, sum = self.uncor_operator(h_p=uncor_neigh, neighbor=uncor_neigh)
                uncor_index_b = uncor_index.float().expand(out.shape[0], self.out_features)
                uncor_out_re = out.real.type(torch.float32)
                uncor_out_im = out.imag.type(torch.float32)

                uncor_out = (uncor_index_b * uncor_out_re) + 1.j*(uncor_index_b * uncor_out_im)
                del uncor_index, uncor_index_a, uncor_index_b, uncor_out_re, uncor_out_im

                out = self.cor_operator(h_p=cor_neigh, sum=sum)
                cor_index_b = cor_index.float().expand(out.shape[0], self.out_features)
                cor_out_re = out.real.type(torch.float32)
                cor_out_im = out.imag.type(torch.float32)

                cor_out = (cor_index_b * cor_out_re) + 1.j*(cor_index_b * cor_out_im)
                del cor_index, cor_index_a, cor_index_b, cor_out_re, cor_out_im

                out = uncor_out + cor_out

                ########### Fatt ###########
                out = self.Fatt(out, ro_theta)
                ############################

                output[index] = out.type(torch.complex128)

        return output


class GNN(nn.Module):
    def __init__(self, meshgrid, in_features, out_features, ring_width, max_neigh, h, w):
        super(GNN, self).__init__()
        self.h = h
        self.w = w
        # self.FGNN1 = FGNN(in_features, out_features, size)
        # self.FGNN2 = FGNN(out_features, out_features, size)
        self.ReLU = ReLU()

        module_list = [FGNN(meshgrid, in_features, out_features, ring_width, max_neigh, h, w),
                       ReLU(),
                       FGNN(meshgrid, out_features, out_features, ring_width, max_neigh, h, w),
                       ReLU(),
                       FGNN(meshgrid, out_features, out_features, ring_width, max_neigh, h, w),]

        self.layer = nn.ModuleList(module_list)

        # feature_extractor = nn.Sequential(
        #     FGNN(1, 16)
        #     nn.ReLU()
        #     FGNN(16, 16)
        #     nn.ReLU()
        #     FGNN(16, 16)
        #     channelwise pooling
        #     multiply mask
        # )
    def forward(self, input_image, input_mask, output):
        # FGNN 1x16 > ReLU > FGNN 16x16 > ReLU > FGNN 16x16 > Ch pooling > multiply mask
        x = self.layer[0](input_image, input_mask, output)
        x = self.layer[1](x)
        x = self.layer[2](x, input_mask, x)
        x = self.layer[3](x)
        x = self.layer[4](x, input_mask, x)

        # Channel-wise mean pooling
        x = torch.mean(x, dim=2)

        # for i in range(len(self.layer)):
        #     x, y = self.layer[i](x, input_mask, y)
        #     if i != len(self.layer)-1:
        #         x = nn.functional.relu(x.real) + 1.j * nn.functional.relu(x.imag)

        x = input_image - x * input_mask
        x = ifftn(ifftshift(x)) * (self.h*self.w)
        x = self.ReLU(x)
        return x

if __name__ == "__main__":
    pass