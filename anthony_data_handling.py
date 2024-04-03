import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple
from common.augmentation import to_coords
import random

class PDEDataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 mode: str,
                 resolution: list=None,
                 augmentation = None,
                 augmentation_ratio: float=0.0,
                 shift: str='fourier',
                 load_all: bool=False,
                 num_samples: int=-1,
                 device: str = 'cuda:0') -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.resolution = (250, 100) if resolution is None else resolution
        self.data = f[self.mode]

        self.num_samples = len(self.data["u"])+2 if(num_samples == -1) else num_samples
        self.u = self.data["u"][:self.num_samples]
        self.length = len(self.u)
        self.alpha = self.data["alpha"][:self.num_samples]
        self.beta = self.data["beta"][:self.num_samples]
        self.gamma = self.data["gamma"][:self.num_samples]

        self.x = torch.tensor(np.array(self.data["x"][:self.num_samples]))
        self.t = torch.tensor(np.array(self.data["t"][:self.num_samples]))

        self.tmin = self.t[0]
        self.tmax = self.t[-1]
        self.nt = len(self.t)
        self.dt = (self.tmax - self.tmin) / self.nt

        self.xmin = self.x[0]
        self.xmax = self.x[-1]
        self.nx = len(self.x)
        self.dx = (self.xmax - self.xmin)/ self.nx
        
        self.augmentation = [] if(augmentation is None) else augmentation 
        self.shift = shift
        self.augmentation_ratio = augmentation_ratio

        self.device = device

        if load_all:
            self.u = torch.tensor(self.u[:]).to(device)
            self.alpha = torch.tensor(self.alpha[:]).to(device)
            self.beta = torch.tensor(self.beta[:]).to(device)
            self.gamma = torch.tensor(self.gamma[:]).to(device)
            self.x = self.x.to(device)
            self.t = self.t.to(device)

            f.close()

    def __len__(self):
        return self.length*(len(self.augmentation)+1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
        t_idx = idx % (len(self.augmentation) + 1)
        idx = idx // (len(self.augmentation) + 1)
        u = self.u[idx]
        x = self.x

        # Base resolution trajectories (numerical baseline) and equation specific parameters
        variables = {}
        variables['alpha'] = self.alpha[idx]
        variables['beta'] = self.beta[idx]
        variables['gamma'] = self.gamma[idx]

        if self.mode == "train" and self.augmentation is not None:
            if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
                t = self.t
                # Augment data
                X = to_coords(x, t)

                if not torch.is_tensor(u):
                    u = torch.tensor(u)

                sol = (u, X)
                sol = self.augmentation[t_idx](sol, self.shift)
                u = sol[0]

        return u, x, variables

        
class PDEDataset2D(Dataset):
    """Load samples of a 2D PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: str,
                 mode: str,
                 resolution: list=None,
                 augmentation = None,
                 augmentation_ratio: float=0.0,
                 shift: str='fourier',
                 load_all: bool=False,
                 device: str='cuda:0',
                 num_samples: int=-1) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE 
            mode: [train, valid, test]
            resolution: base resolution of the dataset [nt, nx, ny]
            augmentation: Data augmentation object
            augmentation_ratio: Probability to augment data
            load_all: load all the data into memory
            device: if load_all, load data onto device
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.resolution = (100, 64, 64) if resolution is None else resolution
        self.data = f[self.mode]
        self.num_samples = len(self.data["u"])+2 if(num_samples == -1) else num_samples
        self.u = self.data["u"][:self.num_samples]
        self.length = len(self.u)
        self.nu = self.data["nu"][:self.num_samples]
        self.ax = self.data["ax"][:self.num_samples]
        self.ay = self.data["ay"][:self.num_samples]
        self.cx = self.data["cx"][:self.num_samples]
        self.cy = self.data["cy"][:self.num_samples]

        self.x = torch.tensor(np.array(self.data["x"][:self.num_samples]))
        self.t = torch.tensor(np.array(self.data["t"][:self.num_samples]))

        self.tmin = self.t[0]
        self.tmax = self.t[-1]
        self.nt = len(self.t)
        self.dt = (self.tmax - self.tmin) / self.nt

        self.xmin = self.x[0, 0, 0]
        self.xmax = self.x[0, 0, -1]
        self.nx = len(self.x[0, 0])
        self.dx = (self.xmax - self.xmin)/ self.nx
        
        self.augmentation = augmentation
        self.shift = shift
        self.augmentation_ratio = augmentation_ratio

        if load_all:
            self.u = torch.tensor(self.u[:]).to(device)
            self.nu = torch.tensor(self.nu[:]).to(device)
            self.ax = torch.tensor(self.ax[:]).to(device)
            self.ay = torch.tensor(self.ay[:]).to(device)
            self.cx = torch.tensor(self.cx[:]).to(device)
            self.cy = torch.tensor(self.cy[:]).to(device)

            self.x = self.x.to(device)
            self.t = self.t.to(device)

            f.close()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        u = self.u[idx]
        x = self.x
        
        variables = {}
        variables['nu'] = self.nu[idx] 
        variables['ax'] = self.ax[idx]
        variables['ay'] = self.ay[idx]
        variables['cx'] = self.cx[idx]
        variables['cy'] = self.cy[idx]

        if self.mode == "train" and self.augmentation is not None:
            if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
                pde = self.get_PDE(variables)
                if not torch.is_tensor(u):
                    u = torch.tensor(u)
                u = self.augmentation(u, pde, self.shift)

        return u, x, variables
    
    def get_PDE(self, variables):
        if variables['ax'] != 0 and variables['ay'] != 0:
            return "advection"
        elif variables["cx"] != 0 and variables["cy"] != 0:
            return "burgers"
        elif variables["nu"] != 0:
            return "heat"
        else:
            raise ValueError("PDE not found")
