import glob

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import gc
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging, pickle, h5py


from .factorization_module import FABlock2D
from .positional_encoding_module import GaussianFourierFeatureTransform
from .basics import PreNorm, MLP, masked_instance_norm
#from utils import Trainer, dict2namespace, index_points, load_checkpoint, save_checkpoint, ensure_dir
import yaml
from torch.optim.lr_scheduler import OneCycleLR
#from loss_fn import rel_l2_loss

from matplotlib import pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
import shutil
from collections import OrderedDict
#from train_utils import CurriculumSampler
import random

torch.backends.cudnn.benchmark = True


class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(2, dim // 2, 8),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
    #def forward(self, u, pos_lst):
        '''
            This might be wrong now...
        '''
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1)
        for pos_enc, attn_layer in self.layers:
            u += pos_enc(pos).view(1, nx, ny, -1)
            #u = attn_layer(u, pos_lst) + u
            u = attn_layer(u, pos_lst).reshape(u.shape) + u
        return u


class FactFormer2D(nn.Module):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.config = config
        # self.resolutions = config.resolutions   # hierachical resolutions, [16, 8, 4]
        # self.out_resolution = config.out_resolution

        self.in_dim = config["in_dim"]
        self.in_tw = config["initial_step"]
        self.out_dim = config["out_dim"]
        self.out_tw = config["t_bundle"]

        self.dim = config["dim"]                 # dimension of the transformer
        self.depth = config["depth"]           # depth of the encoder transformer
        self.dim_head = config["dim_head"]

        self.heads = config["heads"]

        self.pos_in_dim = config["pos_in_dim"]
        self.pos_out_dim = config["pos_out_dim"]
        self.kernel_multiplier = config["kernel_multiplier"]
        self.latent_multiplier = config["latent_multiplier"]
        self.latent_dim = int(self.dim * self.latent_multiplier)
        self.max_latent_steps = config["max_latent_steps"]

        self.channels = self.in_tw*self.in_dim

        # flatten time window
        self.to_in = nn.Linear(self.in_tw*self.in_dim, self.dim, bias=True)

        # assume input is b c t h w d
        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier)
        self.expand_latent = nn.Linear(self.dim, self.latent_dim, bias=False)
        self.latent_time_emb = nn.Parameter(torch.randn(1, self.max_latent_steps,
                                                        1, 1, self.latent_dim) * 0.02,
                                            requires_grad=True)

        self.propagator = PreNorm(self.latent_dim,
                                  MLP([self.latent_dim, self.dim, self.latent_dim], act_fn=nn.GELU()))
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=int(8 * self.latent_multiplier), num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, self.dim, kernel_size=1, stride=1, padding=0,
                      groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim // 2, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,
                u,
                grid,
                latent_steps=1,
                ):
        b, nx, ny, c = u.shape
        u = self.to_in(u)
        pos_lst = [grid[0,:,0,0].unsqueeze(-1), grid[0,0,:,1].unsqueeze(-1)]
        u = self.encoder(u, pos_lst)
        u_lst = []
        u = self.expand_latent(u)
        for l_t in range(latent_steps):
            u = u + self.latent_time_emb[:, l_t, ...]
            u = self.propagator(u) + u
            u_lst.append(u)

        u = torch.cat(u_lst, dim=0)
        u = self.simple_to_out(u)
        u = rearrange(u, '(t b) c (nx ny) -> b t nx ny c', nx=nx, ny=ny, t=latent_steps)
        return u.permute(0,2,3,1,4)[...,0,:]


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
        #self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        out_size = int((self.img_size[0] + 2*0 - 1*(patch_size[0] - 1) - 1)/stride_size[0] + 1)
        #print("\nOUT SIZE: {}\n".format(out_size))
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


class LLMFactFormer2D(nn.Module):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.config = config
        # self.resolutions = config.resolutions   # hierachical resolutions, [16, 8, 4]
        # self.out_resolution = config.out_resolution

        self.in_dim = config["in_dim"]
        self.in_tw = config["initial_step"]
        self.out_dim = config["out_dim"]
        self.out_tw = config["t_bundle"]

        self.dim = config["dim"]                 # dimension of the transformer
        self.depth = config["depth"]           # depth of the encoder transformer
        self.dim_head = config["dim_head"]

        self.heads = config["heads"]

        self.pos_in_dim = config["pos_in_dim"]
        self.pos_out_dim = config["pos_out_dim"]
        self.kernel_multiplier = config["kernel_multiplier"]
        self.latent_multiplier = config["latent_multiplier"]
        self.latent_dim = int(self.dim * self.latent_multiplier)
        self.max_latent_steps = config["max_latent_steps"]

        self.llm = config['llm']
        self.img_size = config['img_size']
        self.channels = self.in_tw*self.in_dim

        # flatten time window
        self.to_in = nn.Linear(self.in_tw*self.in_dim, self.dim, bias=True)

        # assume input is b c t h w d
        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier)
        #self.expand_latent = nn.Linear(self.dim, self.latent_dim, bias=False)
        self.expand_latent = nn.Linear(self.dim+1, self.latent_dim, bias=False)
        self.latent_time_emb = nn.Parameter(torch.randn(1, self.max_latent_steps,
                                                        1, 1, self.latent_dim) * 0.02,
                                            requires_grad=True)

        self.propagator = PreNorm(self.latent_dim,
                                  MLP([self.latent_dim, self.dim, self.latent_dim], act_fn=nn.GELU()))
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=int(8 * self.latent_multiplier), num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, self.dim, kernel_size=1, stride=1, padding=0,
                      groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim // 2, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.clip_embedding_layer = PatchEmbed(img_size=self.img_size, patch_size=16, in_chans=self.dim,
                                               embed_dim=self.dim, out_dim=self.dim, act='gelu')
        # Hard-coded For CLIP
        input_embed_dim = 768 if(self.llm == 'all-mpnet-base-v2') else 384
        self.sentence_proj = nn.Sequential(
                                 nn.Linear(input_embed_dim, 2*self.dim),
                                 nn.ReLU(),
                                 nn.Linear(2*self.dim, 2*self.dim),
                                 nn.ReLU(),
                                 nn.Linear(2*self.dim, self.dim),
                                 nn.ReLU(),
                                 #nn.Linear(self.dim, self.dim//2)
                                 nn.Linear(self.dim, self.img_size)
        )

        self.x_channel_proj = nn.Sequential(
                                 nn.Linear(15**2, self.dim),
                                 nn.ReLU(),
                                 nn.Linear(self.dim, self.dim),
                                 nn.ReLU(),
                                 nn.Linear(self.dim, 1)
        )
        self.emb_proj = nn.Sequential(
                                 nn.Linear(self.dim, self.dim),
                                 nn.ReLU(),
                                 #nn.Linear(self.dim, self.dim//2)
                                 nn.Linear(self.dim, self.img_size)
        )

    def forward(self,
                u,
                grid,
                latent_steps=1,
                sentence_embeddings=None,
                clip=False,
                return_embedding=False,
                ep=1
                ):
        b, nx, ny, c = u.shape

        # Get encoded value
        u = self.to_in(u)
        pos_lst = [grid[0,:,0,0].unsqueeze(-1), grid[0,0,:,1].unsqueeze(-1)]
        u = self.encoder(u, pos_lst)

        # Get embeddings for CLIP
        u_emb = self.clip_embedding_layer(u.permute(0,3,1,2)).flatten(2,3)
        u_emb = self.emb_proj(self.x_channel_proj(u_emb)[...,0])
        sentence_emb = self.sentence_proj(sentence_embeddings)
        #print(u.shape)
        #print(u_emb.shape, self.clip_embedding_layer.num_patches)
        #print(sentence_embeddings.shape)
        #print()
        #print(self.sentence_proj(sentence_embeddings).shape)
        #print(self.emb_proj(self.x_channel_proj(u_emb)[...,0]).shape)
        #print()
        #print()

        if(return_embedding):
            cross_corr = u_emb @ sentence_emb.T
            return torch.cat((sentence_emb.unsqueeze(-1), u_emb.unsqueeze(-1)), dim=-1), cross_corr
        if(clip):
            cross_corr = u_emb @ sentence_emb.T
            return cross_corr
        else:
            #print()
            #print(u_emb.shape, sentence_emb.shape)
            embedding = torch.bmm(u_emb.unsqueeze(2), sentence_emb.unsqueeze(1)).unsqueeze(-1)
            #print()
            #embedding = torch.cat((u_emb, sentence_emb), dim=-1)[:,None,None,:]
            print(u.shape, embedding.shape)
            u = torch.cat((u, embedding), dim=-1)

        u_lst = []
        u = self.expand_latent(u)
        for l_t in range(latent_steps):
            u = u + self.latent_time_emb[:, l_t, ...]
            u = self.propagator(u) + u
            u_lst.append(u)

        u = torch.cat(u_lst, dim=0)
        u = self.simple_to_out(u)
        u = rearrange(u, '(t b) c (nx ny) -> b t nx ny c', nx=nx, ny=ny, t=latent_steps)
        return u.permute(0,2,3,1,4)[...,0,:]

    
    def finished_pretraining(self):
        pass

