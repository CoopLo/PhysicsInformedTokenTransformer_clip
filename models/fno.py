#!/usr/bin/env python3

"""
FNO. Implementation taken and modified from
https://github.com/zongyi-li/fourier_neural_operator

MIT License

Copyright (c) 2020 Zongyi Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample

#import torch_dct as dct

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, num_channels, modes=16, width=64, initial_step=10, dropout=0.1):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step*num_channels+1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        #x = F.silu(x)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        #x = F.silu(x)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        #x = F.silu(x)
        x = self.dropout(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        #x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)

    def get_loss(self, x, y, grid, loss_fn):
        y_pred = self.forward(x, grid)[...,0,0]
        #print(y_pred.shape)
        #print(y.shape)
        return y_pred, loss_fn(y_pred, y)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10, dropout=0.1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.channels = num_channels*initial_step
        self.fc0 = nn.Linear(initial_step*num_channels+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)

        # TODO: Make this robust to not choosing coefficient conditioning information
        self.fc2 = nn.Linear(128, num_channels)
        #self.fc2 = nn.Linear(128, num_channels-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


    def get_loss(self, x, y, grid, loss_fn):
        y_pred = self.forward(x, grid)[...,0,0]
        return y_pred, loss_fn(y_pred, y)
    

# Take from DPOT code
ACTIVATION = {
        'gelu':nn.GELU(),
        'tanh':nn.Tanh(),
        'sigmoid':nn.Sigmoid(),
        'relu':nn.ReLU(),
        'leaky_relu':nn.LeakyReLU(0.1),
        'softplus':nn.Softplus(),
        'ELU':nn.ELU(),
        'silu':nn.SiLU()
}                
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128,act='gelu'):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        img_size = (img_size, img_size)
                 
        stride_size = (patch_size//2, patch_size//2)

        patch_size = (patch_size, patch_size) 
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Need to update for overlapping patches...
        out_size = int((self.img_size[0] + 2*0 - 1*(patch_size[0] - 1) - 1)/stride_size[0] + 1)
        self.out_size = (out_size, out_size)

        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )
        

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
               f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class CLIPFNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10, dropout=0.1, embed_dim=32, llm_dim=384):
        super(CLIPFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.channels = num_channels*initial_step

        self.fc0 = nn.Linear(initial_step*num_channels+2, self.width)
        #self.fc0 = nn.Linear(initial_step*num_channels+2+1, self.width)
        #self.fc0_finetune = nn.Linear(initial_step*num_channels+2+1, self.width)

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        #self.conv0 = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2)
        #self.conv0_finetune = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2)
        self.conv0_finetune = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        #self.w0 = nn.Conv2d(self.width+1, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        #self.fc1 = nn.Linear(self.width, 128)
        self.fc1 = nn.Linear(self.width+1, 128)
        self.fc2 = nn.Linear(128, num_channels)
        #self.fc2 = nn.Linear(128, num_channels-5)
        self.dropout = nn.Dropout(dropout)

        # Patch embedding
        #self.to_patch_embedding = PatchEmbed(img_size=128, patch_size=8, in_chans=self.width,
        #                                    embed_dim=1, out_dim=1, act='gelu')
        #self.to_patch_embedding = PatchEmbed(img_size=130, patch_size=8, in_chans=self.width,
        #                                    embed_dim=1, out_dim=1, act='gelu')
        #self.to_patch_embedding = PatchEmbed(img_size=66, patch_size=8, in_chans=self.width,
        #                                    embed_dim=1, out_dim=1, act='gelu')
        self.to_patch_embedding = PatchEmbed(img_size=34, patch_size=8, in_chans=self.width,
                                            embed_dim=1, out_dim=1, act='gelu')

        self.sentence_proj = nn.Sequential(
                                 #nn.Linear(384, embed_dim),
                                 nn.Linear(llm_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )
        self.x_proj = nn.Sequential(
                                 #nn.Linear(31*31, embed_dim),
                                 #nn.Linear(15**2, embed_dim),
                                 nn.Linear(7**2, embed_dim),

                                 #nn.Linear(32**3, embed_dim),
                                 #nn.Linear((initial_step*num_channels+2)*32*32, embed_dim),
                                 #nn.Linear(32*34*34, embed_dim),
                                 #nn.Linear(32**2, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )
        self.proj_up = nn.Sequential(
                                 nn.Linear(32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 #nn.Linear(256, 128*128)
                                 #nn.Linear(256, 130*130)
                                 #nn.Linear(256, 66*66)
                                 nn.Linear(256, 34*34)
        )
        self.pretrained = False


    def forward(self, x, grid, sentence_embeddings, clip=False, return_embedding=False):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)

        # During pretraining get embedding from model output
        #print(x.shape)
        #print(x_proj)
        #print(x.shape)
        #x_emb = self.x_proj(x.flatten(1,3))

        #raise
        sentence_emb = self.sentence_proj(sentence_embeddings)
        #if(return_embedding):
        #    return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1)
        #if(clip):
        #    cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
        #    return cross_corr
        #else:
        #    embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach()
        #    embedding = self.proj_up(embedding)
        #    x = torch.cat((x, embedding.reshape((x.shape[0], 32, 32, 1))), dim=-1)

        x = self.fc0(x)
        #print(x.shape)

        x = x.permute(0, 3, 1, 2)
        #x_emb = self.x_proj(self.to_patch_embedding(x).flatten(1,3))

        #if(return_embedding):
        #    cross_corr = x_emb @ sentence_emb.T
        #    return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        #if(clip):
        #    cross_corr = x_emb @ sentence_emb.T
        #    return cross_corr
        #else:
        #    embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach()
        #    embedding = self.proj_up(embedding)
        #    x = torch.cat((x, embedding.reshape((x.shape[0], 1, 128, 128))), dim=1)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x_emb = self.x_proj(self.to_patch_embedding(x).flatten(1,3))

        if(return_embedding):
            cross_corr = x_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1), cross_corr
        if(clip):
            cross_corr = x_emb @ sentence_emb.T
            return cross_corr
        else:
            embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1)#.detach()
            embedding = self.proj_up(embedding)
            #x = torch.cat((x, embedding.reshape((x.shape[0], 1, 130, 130))), dim=1)
            #x = torch.cat((x, embedding.reshape((x.shape[0], 1, 66, 66))), dim=1)
            x = torch.cat((x, embedding.reshape((x.shape[0], 1, 34, 34))), dim=1)
        #raise

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)


    def get_loss(self, x, y, grid, sentence_embeddings, loss_fn):
        y_pred = self.forward(x, grid, sentence_embeddings)[...,0,0]
        return y_pred, loss_fn(y_pred, y)


    def finished_pretraining(self):
        self.pretrained = True


class LLMFNO2d(nn.Module):
    def __init__(self, num_channels, modes1=12, modes2=12, width=20, initial_step=10, dropout=0.1, embed_dim=32, llm_dim=384,
                 llm=None):
        super(LLMFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        if(llm is not None):
            self.llm = SentenceTransformer(llm, device='cuda')

        self.fc0 = nn.Linear(initial_step*num_channels+2, self.width)
        #self.fc0 = nn.Linear(initial_step*num_channels+2+1, self.width)

        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        #self.conv0 = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2)
        #self.conv1 = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2)
        #self.conv2 = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2)

        # Choose whether or not to add conditioning information here.
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)  # No
        #self.conv3 = SpectralConv2d_fast(self.width+1, self.width, self.modes1, self.modes2) # Yes

        #self.w0 = nn.Conv2d(self.width+1, self.width, 1)
        #self.w1 = nn.Conv2d(self.width+1, self.width, 1)
        #self.w2 = nn.Conv2d(self.width+1, self.width, 1)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Choose whether or not to add conditioning information here.
        self.w3 = nn.Conv2d(self.width, self.width, 1)  # No
        #self.w3 = nn.Conv2d(self.width+1, self.width, 1) # Yes

        self.fc1 = nn.Linear(self.width+1, 128)
        self.fc2 = nn.Linear(128, num_channels)
        self.dropout = nn.Dropout(dropout)

        self.sentence_proj = nn.Sequential(
                                 #nn.Linear(384, embed_dim),
                                 nn.Linear(llm_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )
        self.x_proj = nn.Sequential(
                                 #nn.Linear(32**3, embed_dim),
                                 nn.Linear((initial_step*num_channels+2)*32*32, embed_dim),
                                 #nn.Linear(32*34*34, embed_dim),
                                 #nn.Linear(32**2, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim),
                                 nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim//2)
        )
        self.proj_up = nn.Sequential(
                                 nn.Linear(32, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 #nn.Linear(256, 32*32)
                                 nn.Linear(256, 34*34)
        )

    @torch.enable_grad()
    def _llm_forward(self, sentence):
        tokenized_sentences = self.llm.tokenize(sentence)
        output = self.llm(tokenized_sentences)['sentence_embedding']
        return output

    def forward(self, x, grid, sentence_embeddings, clip=False, return_embedding=False):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)

        ### During pretraining get embedding from model output
        ##x_emb = self.x_proj(x.flatten(1,3))

        ### Embed and project sentences
        ##sentence_emb = self._llm_forward(sentence_embeddings)
        ##sentence_emb = self.sentence_proj(sentence_emb)
        ##if(return_embedding):
        ##    return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1)
        ##if(clip):
        ##    cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
        ##    return cross_corr

        ### Get embedding
        ##embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1).detach()
        ##embedding = self.proj_up(embedding).reshape((x.shape[0], 1, 34, 34))

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x = torch.cat((x, embedding), dim=1) # Add conditioning information to every layer
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x = torch.cat((x, embedding), dim=1) # Add conditioning information to every layer
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x = torch.cat((x, embedding), dim=1) # Add conditioning information to every layer
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)

        x = torch.cat((x, embedding), dim=1) # Add conditioning information to every layer
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # During pretraining get embedding from model output
        print()
        print(x.shape, x.flatten(1,3).shape)
        print()
        print("HERE")
        x_emb = self.x_proj(x.flatten(1,3))

        # Embed and project sentences
        sentence_emb = self._llm_forward(sentence_embeddings)
        sentence_emb = self.sentence_proj(sentence_emb)
        if(return_embedding):
            return torch.cat((sentence_emb.unsqueeze(-1), x_emb.unsqueeze(-1)), dim=-1)
        if(clip):
            cross_corr = torch.bmm(x_emb.unsqueeze(2), sentence_emb.unsqueeze(1))
            return cross_corr

        raise

        # Get embedding
        embedding = torch.cat((x_emb, sentence_emb), dim=-1).unsqueeze(1).detach()
        embedding = self.proj_up(embedding).reshape((x.shape[0], 1, 34, 34))

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)


    def get_loss(self, x, y, grid, sentence_embeddings, loss_fn):
        y_pred = self.forward(x, grid, sentence_embeddings)[...,0,0]
        return y_pred, loss_fn(y_pred, y)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, num_channels, modes1=8, modes2=8, modes3=8, width=20, initial_step=10):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step*num_channels+3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, x3, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)
