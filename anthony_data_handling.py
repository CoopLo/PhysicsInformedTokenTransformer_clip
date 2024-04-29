import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple
#from common.augmentation import to_coords
import random
from sentence_transformers import SentenceTransformer, InputExample
from tqdm import tqdm

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

        # Use LLM if CLIP
        self.clip = clip
        if(self.clip):
            self.sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            self.sentence_embeddings = []
            for idx in range(self.x.shape[0]):
                print(idx)
            raise


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
                 num_samples: int=-1,
                 clip: bool=False,
                 llm: str=None,
                 sentence: bool=False,
                 downsample: int=1) -> None:
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
        self.downsample = downsample
        self.resolution = (100, 64, 64) if resolution is None else resolution
        self.llm = llm
        self.data = f[self.mode]
        if(mode == 'train'):
            self.num_samples = len(self.data["u"])+2 if(num_samples == -1) else num_samples
        else:
            self.num_samples = 768 # Use entire validation set

        idxs = torch.randperm(self.data["u"].shape[0])[:self.num_samples].cpu().numpy()

        self.u = self.data["u"][:][idxs][...,::self.downsample, ::self.downsample]
        self.length = len(self.u)
        self.nu = torch.Tensor(self.data["nu"][:][idxs])
        self.ax = torch.Tensor(self.data["ax"][:][idxs])
        self.ay = torch.Tensor(self.data["ay"][:][idxs])
        self.cx = torch.Tensor(self.data["cx"][:][idxs])
        self.cy = torch.Tensor(self.data["cy"][:][idxs])

        #print()
        #print("NU: {}\t{}".format(self.nu.max(), self.nu.min()))
        #print("AX: {}\t{}".format(self.ax.max(), self.ax.min()))
        #print("AY: {}\t{}".format(self.ay.max(), self.ay.min()))
        #print("CX: {}\t{}".format(self.cx.max(), self.cx.min()))
        #print("CY: {}\t{}".format(self.cy.max(), self.cy.min()))
        #print()
        #raise

        self.x = torch.tensor(np.array(self.data["x"][:]))[...,::self.downsample, ::self.downsample]
        self.t = torch.tensor(np.array(self.data["t"][:]))

        #self.u = self.data["u"][:self.num_samples]
        #self.length = len(self.u)
        #self.nu = self.data["nu"][:self.num_samples]
        #self.ax = self.data["ax"][:self.num_samples]
        #self.ay = self.data["ay"][:self.num_samples]
        #self.cx = self.data["cx"][:self.num_samples]
        #self.cy = self.data["cy"][:self.num_samples]

        #self.x = torch.tensor(np.array(self.data["x"][:self.num_samples]))
        #self.t = torch.tensor(np.array(self.data["t"][:self.num_samples]))

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

        f.close()

        # Use LLM if CLIP
        self.clip = clip
        self.sentence = sentence
        if(self.clip or self.sentence):

            # Only get sentence_embedder if we're not returning whole sentences
            if(self.llm is not None and not self.sentence):
                #self.sentence_embedder = SentenceTransformer(self.llm, device='cpu')
                print("LOADING LLM TO GPU")
                self.sentence_embedder = SentenceTransformer(self.llm, device='cuda')
            elif(not self.sentence):
                self.sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

            self.sentence_embeddings = []
            self.sentences = []
            print("Getting sentence embeddings...")
            for idx in tqdm(range(self.u.shape[0])):
                # Burgers
                if(self.nu[idx] != 0 and self.cx[idx] != 0 and self.cy[idx] != 0):
                    ratio = ((self.cx[idx] + self.cy[idx])**2)**(0.5)/self.nu[idx]
                    #ratios.append(ratio)
                    sentence = 'Burgers equation models a conservative system that can develop shock wave discontinuities.'
                    sentence += ' Burgers equation is a first order quasilinear hyperbolic partial differential equation.'
                    sentence += ' In this case, the advection term has a coefficient of {} in the x direction, {} in the y direction, and the diffusion term has a coefficient of {}.'.format(self.cx[idx], self.cy[idx], self.nu[idx])

                    cls = 'strongly' if(ratio > 100) else 'weakly'
                    sim = ' not ' if(ratio > 100) else ' '
                    sentence += ' This system is {} advection dominanted and does{}behave similarly to heat equation.'.format(cls, sim)
                    sentence += ' The predicted state should look like the input but shifted in space.'
                # Advection
                elif(self.ax[idx] != 0 and self.ay[idx] != 0):
                    adv = ((self.ax[idx] + self.ay[idx])**2)**(0.5)
                    #advs.append(adv)
                    sentence = 'The Advection equation models bulk transport of a substance or quantity. It does not develop shocks.'
                    sentence += ' The Advection equation is a linear hyperbolic partial differential equation.'
                    sentence += ' In this case, the advection term has a coefficient of {} in the x direction, {} in the y direction.'.format(self.ax[idx], self.ay[idx])

                    cls = 'strongly' if(adv > 2) else 'weakly'
                    sentence += ' This system is {} advective.'.format(cls)
                    sentence += ' Ths predicted state should have shocks.' if(cls == 'strongly') else ' The predicted state should look smoother than the inputs'
                # Heat
                elif(self.nu[idx] != 0 and self.cx[idx] == 0 and self.cy[idx] == 0):
                    sentence = 'The Heat equation models how a quantity such as heat diffuses through a given region.'
                    sentence += ' The Heat equation is a linear parabolic partial differential equation.'
                    sentence += ' In this case, the diffusion term has a coefficient of {}.'.format(self.nu[idx])

                    cls = 'strongly' if(self.nu[idx] > 0.01) else 'weakly'
                    sentence += ' This system is {} diffusive.'.format(cls)
                    sentence += ' The predicted state should look smoother than the inputs.'

                sentence += " Give me an embedding that is useful for numerically predicting the target state."
                if(self.sentence):
                    while(len(sentence) < 650): # Pad them to have same length
                        sentence += ' '
                    if(len(sentence) > 650):
                        print(len(sentence))
                        raise
                    self.sentences.append(sentence)
                    #self.sentences.append(InputExample(texts=[sentence], label=0.0))
                else:
                    self.sentence_embeddings.append(self.sentence_embedder.encode(sentence))
            print("Done.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor]:
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

        if(self.clip):
            return u, x.permute(1,2,0), variables, self.sentence_embeddings[idx]
        elif(self.sentence):
            return u, x.permute(1,2,0), variables, self.sentences[idx]
        else:
            return u, x.permute(1,2,0), variables
    
    def get_PDE(self, variables):
        if variables['ax'] != 0 and variables['ay'] != 0:
            return "advection"
        elif variables["cx"] != 0 and variables["cy"] != 0:
            return "burgers"
        elif variables["nu"] != 0:
            return "heat"
        else:
            raise ValueError("PDE not found")
