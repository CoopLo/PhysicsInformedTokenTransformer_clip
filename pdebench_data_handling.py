# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
  Authors:  Timothy Praditia (timothy.praditia@iws.uni-stuttgart.de)
            Raphael Leiteritz (raphael.leiteritz@ipvs.uni-stuttgart.de)
            Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (makoto.takamoto@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class FNODatasetSingle(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1,

                 clip = False,
                 llm = None,
                 sentence = False,
                 bcs = False,
                 coeff = False,
                 qualitative = False,

                 max_timestep = -1,
                 
                 image_size = None,

                 transfer = False,
                 normalize = False
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.filename = filename
        self.saved_folder = saved_folder
        self.llm = llm
        self.sentence = sentence
        self.coeff = coeff
        self.qualitative = qualitative
        self.transfer = transfer
        self.bcs = bcs
        self.normalize = normalize

        # LLM attributes
        self.clip = clip
        self.llm = llm
        self.sentence = sentence
        self.qualitative = qualitative

        # Get coefficient information. Inidicator for SW and RD (easier to always construct and return then handle later)
        self.get_coeffs()

        root_path = os.path.join(os.path.abspath(saved_folder), filename)

        # Get train/test slice first. TODO No longer supports -1
        # Pretraining must have num_samples < finetune num_samples otherwise there are data leaks.
        #if num_samples_max>0:
        #    num_samples_max  = min(num_samples_max, self.data.shape[0])
        #else:
        #    num_samples_max = self.data.shape[0]
        test_idx = int(num_samples_max * test_ratio)
        idxs = slice(0, test_idx, 1) if(if_test) else slice(test_idx, num_samples_max, 1)

        if filename[-2:] != 'h5':
            print(f".HDF5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    #_data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                    _data = np.array(f['density'][idxs], dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if(image_size is not None):
                        old_rr = reduced_resolution
                        reduced_resolution = idx_cfd[2]//image_size
                        print("OVERRIDING REDUCED RESOLUTION FROM {} TO {}".format(old_rr, reduced_resolution))

                    if len(idx_cfd)==3:  # 1D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              3],
                                            dtype=np.float32)
                        #density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(idx_cfd)==4:  # 2D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              4],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch

                        # pressure
                        #_data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = np.array(f['pressure'][idxs], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch

                        # Vx
                        #_data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = np.array(f['Vx'][idxs], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        # Vy
                        #_data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = np.array(f['Vy'][idxs], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch

                        # Grid
                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

                        # TODO: Time has 22 steps but everything else has 21, so remove final timestep
                        self.tsteps_t = np.array(f['t-coordinate'])[::reduced_resolution_t][:-1] 
                
                    if len(idx_cfd)==5:  # 3D
                        self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                              idx_cfd[2]//reduced_resolution,
                                              idx_cfd[3]//reduced_resolution,
                                              idx_cfd[4]//reduced_resolution,
                                              mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                              5],
                                             dtype=np.float32)
                        # density
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                                                                    
                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        elif filename[-2:] == 'h5':  # SWE-2D (RDB)
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                if(image_size is not None):
                    old_rr = reduced_resolution
                    reduced_resolution = data_arrays[0].shape[2]//image_size
                    print("OVERRIDING REDUCED RESOLUTION FROM {} TO {}".format(old_rr, reduced_resolution))

                # Doesn't appear to be an easier way to do this...
                #_data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))[idxs]   # [batch, nt, nx, ny, nc]

                _data = _data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                _grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data.cuda()
                self.grid = _grid.cuda()
                self.tsteps_t = tsteps_t.cuda()

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)
        self.tsteps_t = torch.tensor(self.tsteps_t)

        self.dt = self.tsteps_t[1] - self.tsteps_t[0] * torch.ones(len(self.tsteps_t)).unsqueeze(0)

        # Need to add empty data channels when doing transfer learning because base model has 4
        ds = self.data.shape
        if(self.transfer and ds[-1] != 4):
            self.data = torch.cat((self.data, torch.zeros(ds[0], ds[1], ds[2], ds[3], 4-ds[-1])), dim=-1)

        # Add sentence embeddings
        if(self.clip or self.sentence):
            self.get_llm()
            self.sentence_embeddings = self.sentence_embeddings.cuda()

        self.data = self.data.cuda()
        self.grid = self.grid.cuda()
        #print()
        #print(self.data.shape)
        if(self.normalize):
            self.means = self.data.mean(dim=(0,1,2))[None,None,None]
            self.stds = self.data.std(dim=(0,1,2))[None,None,None]
            #print(self.data.shape)
            #print(self.means.shape)
            #print(self.stds.shape)
            #print()
            #print(self.data.min(), self.data.max())
            #print(self.means.max(), self.means.min())
            #print(self.stds.max(), self.stds.min())
            #print(self.means)
            #print(self.stds)
            self.data = (self.data - self.means)/self.stds
            #print(self.data.min(), self.data.max())
        #print()
        #raise


    def get_coeffs(self):
        if('rdb' in self.filename):
            self.coeffs = [1., 0., 0., 0., 0.]
        elif('diff-react' in self.filename):
            self.coeffs = [0., 1., 0., 0., 0.]
        elif('CFD' in self.filename):
            #print("\n\nFILE: {}\n\n".format(self.filename))
            sf = self.filename.split("_")
            self.coeffs = [0., 0., float(sf[3][1:]), float(sf[4][3:]), float(sf[5][4:])]
        self.coeffs = torch.Tensor(self.coeffs)


    def get_llm(self):
        # Only get sentence_embedder if we're not returning whole sentences
        if(self.llm is not None and not self.sentence):
            print("LOADING LLM TO GPU")
            self.sentence_embedder = SentenceTransformer(self.llm, device='cuda')
        elif(not self.sentence):
            self.sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')

        self.sentence_embeddings = []
        self.sentences = []

        # Create sentences
        print("\nGetting sentence embeddings...")
        for idx in tqdm(range(len(self.data))):
            if('shallow-water' in self.saved_folder):
                # Equation description
                sentence = "The Shallow-Water equations are a set of hyperbolic partial differential equations "
                sentence += "that describe the flow below a pressure surface in a fluid. "

                # Boundary conditions
                if(self.bcs):
                    sentence += "This system has homogeneous Neumann boundary conditions "
                    sentence += "with a derivative of 0 at the boundary."
                if(self.coeff):
                    pass
                if(self.qualitative):
                    sentence += " This system simulates a radial dam break. Waves propagate outward in a circular pattern."

            elif('diffusion-reaction' in self.saved_folder):
                # Equation description
                sentence = "The Diffusion-Reaction equations model the change in concentration of one or more chemical "
                sentence += "substances, in which the substances are transformed into each other, "
                sentence += "and diffusion which causes the substances to spread out over a surface in space. " 

                # Boundary conditions
                if(self.bcs):
                    sentence += "This system has no-flow Neumann boundary conditions "
                    sentence += "with a derivative of 0 at the boundary."
                if(self.coeff):
                    pass
                if(self.qualitative):
                    pass

            elif('2D_Train' in self.saved_folder):
                # Equation description
                sentence = "The compressible Navier-Stokes equations model the motion of fluids with variable density. "

                # Boundary conditions
                if(self.bcs):
                    sentence += "This system has periodic boundary conditions. "

                if(self.coeff):
                    split_fname = self.filename.split("_")
                    sentence += "In this case, the mach number is {}, ".format(float(split_fname[3][1:]))
                    sentence += "the shear viscosity is {}, ".format(float(split_fname[4][3:]))
                    sentence += "and the bulk viscosity is {}.".format(float(split_fname[5][4:]))
                if(self.qualitative):
                    if("Rand" in self.saved_folder):
                        sentence += " A random field is the initial condition, which is not turbulent."
                    elif("Turb" in self.saved_folder):
                        sentence += " A turbulent velocity field is the initial condition."
                        sentence += " This will produce chaotic flow patterns with small scale structure."
                    else:
                        raise ValueError("Unexpected dataset: {}/{}".format(self.saved_folder, self.filename))
            else:
                raise NotImplementedError("Only shallow_water and diffusion_reaction are supported for now.")


            if(self.sentence):
                raise NotImplementedError("Need to implement full LLM training.")
            else:
                self.sentence_embeddings.append(self.sentence_embedder.encode(sentence))

        self.sentence_embeddings = torch.Tensor(np.array(self.sentence_embeddings))
        del self.sentence_embedder
        print("Done.")
            

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #TODO: Different training strategies, currently...?
        #return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid
        if(self.sentence or self.clip):
            return self.data[idx], self.grid, self.coeffs, self.dt[0], self.sentence_embeddings[idx] # Grid and coeff is constant
        else:
            return self.data[idx], self.grid, self.coeffs, self.dt[0] # Grid is constant


class FNODatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        
        # Define path to files
        self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        
        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1-test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])
        
        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]
        
            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)
            
            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1,len(data.shape)-1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)
            
            # Extract spatial dimension of data
            dim = len(data.shape) - 2                                               
            
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                grid = torch.stack((X,Y),axis=-1)
            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                z = np.array(seed_group["grid"]["z"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                grid = torch.stack((X,Y,Z),axis=-1)
        
        return data[...,:self.initial_step,:], data, grid


class MultiDataset(Dataset):
    def __init__(self, filenames,
                 initial_step=10,
                 saved_folders=[],
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1,

                 clip = False,
                 llm = None,
                 sentence = False,
                 coeff = False,
                 qualitative = False,

                 image_size = None,
                 bcs = False,
                 normalize = False,
                 ):

        #self.dsets = []
        self.dsets, self.data, self.grids, self.dt, self.coeff = [], [], [], [], []

        self.clip = clip
        self.sentence = sentence

        for fname, saved_folder in zip(filenames, saved_folders):
            print(fname, saved_folder)
            self.dsets.append(FNODatasetSingle(
                    filename=fname,
                    saved_folder=saved_folder,
                    initial_step=initial_step,
                    reduced_resolution=reduced_resolution,
                    reduced_resolution_t=reduced_resolution_t,
                    reduced_batch=reduced_batch,
                    if_test=if_test,
                    test_ratio=test_ratio,
                    num_samples_max=num_samples_max,

                    clip=clip,
                    llm=llm,
                    bcs=bcs,
                    sentence=sentence,
                    coeff=coeff,
                    qualitative=qualitative,

                    image_size=image_size,

                    normalize=normalize,
            ))

            # Need to adjust resolution and timeframe... try down first, then try up if its an issue, I suppose
            #TODO See if interpolating is any better.
            #print()
            #print(self.dsets[-1].data.shape)
            #print()
            if(self.dsets[-1].data.shape[-1] == 1):  # Shallow water case
                shape = list(self.dsets[-1].data.shape)
                shape[-1] = 3
                data = torch.cat((self.dsets[-1].data, torch.zeros(shape)), dim=-1)
                data = data[:,:,:,:21]
                self.data.append(data)

            elif(self.dsets[-1].data.shape[-1] == 2): # Works because we only need to go to 4
                data = torch.cat((self.dsets[-1].data, torch.zeros(self.dsets[-1].data.shape)), dim=-1)
                data = data[:,:,:,:21]
                self.data.append(data)

            else:  # CFD case
                if(self.dsets[-1].data.shape[1] in [256, 512]):
                    self.data.append(self.dsets[-1].data[:,::4,::4,:])
                else:
                    self.data.append(self.dsets[-1].data)

            if(self.dsets[-1].grid.shape[0] in [256, 512]):
                self.grids.append(self.dsets[-1].grid[::4,::4].unsqueeze(0))
            else:
                self.grids.append(self.dsets[-1].grid.unsqueeze(0))

            self.dt.append(self.dsets[-1].dt[:,:21])
            self.coeff.append(self.dsets[-1].coeffs.unsqueeze(0))

        self.dt = torch.cat(self.dt, dim=0).cuda()
        self.data = torch.cat(self.data, dim=0).cuda()
        self.grids = torch.cat(self.grids, dim=0).cuda()
        self.coeff = torch.cat(self.coeff, dim=0).cuda()
        print(self.dt.shape, self.data.shape, self.grids.shape, self.coeff.shape)


    def __len__(self):
        #return sum(len(d) for d in self.dsets)
        return len(self.data)

    def __getitem__(self, idx):
        dset_idx = idx%len(self.dsets)
        sample_idx = idx//len(self.dsets)
        if(self.clip or self.sentence):
            return self.data[idx], \
                   self.grids[dset_idx], \
                   self.coeff[dset_idx], \
                   self.dt[dset_idx], \
                   self.dsets[dset_idx].sentence_embeddings[sample_idx]
        else:
            return self.data[idx], \
                   self.grids[dset_idx], \
                   self.coeff[dset_idx], \
                   self.dt[dset_idx]
