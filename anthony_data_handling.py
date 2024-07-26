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
                 downsample: int=1,
                 debug: bool=False,
                 subset: str='heat,adv,burger',
                 coeff: bool=True,
                 qualitative: bool=False
                 ) -> None:
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
        self.debug = debug
        self.subset = subset
        self.device = device
        self.data = f[self.mode]

        # Different embedding strategies (Maybe rename to be more clear...)
        self.clip = clip                # Use LLM
        self.sentence = sentence        # Return sentences and train LLM end-to-end
        self.coeff = coeff              # Use sentence information
        self.qualitative = qualitative  # Include qualitative information

        if(mode == 'train'):
            self.num_samples = len(self.data["u"])+2 if(num_samples == -1) else num_samples
        else:
            self.num_samples = 50 if(self.debug) else 768 # Use entire validation set
        #idxs = torch.randperm(self.data["u"].shape[0])[:self.num_samples].cpu().numpy()

        # Get data
        #self.u = self.data["u"][:][idxs][...,::self.downsample, ::self.downsample]
        self.u = torch.Tensor(self.data["u"][:][...,::self.downsample, ::self.downsample]).to(self.device)
        
        # Get coefficients
        #self.nu = torch.Tensor(self.data["nu"][:][idxs])
        #self.ax = torch.Tensor(self.data["ax"][:][idxs])
        #self.ay = torch.Tensor(self.data["ay"][:][idxs])
        #self.cx = torch.Tensor(self.data["cx"][:][idxs])
        #self.cy = torch.Tensor(self.data["cy"][:][idxs])
        self.nu = torch.Tensor(self.data["nu"][:]).to(self.device)
        self.ax = torch.Tensor(self.data["ax"][:]).to(self.device)
        self.ay = torch.Tensor(self.data["ay"][:]).to(self.device)
        self.cx = torch.Tensor(self.data["cx"][:]).to(self.device)
        self.cy = torch.Tensor(self.data["cy"][:]).to(self.device)
        self.coeffs = torch.cat((self.nu.unsqueeze(0), self.ax.unsqueeze(0),
                                 self.ay.unsqueeze(0), self.cx.unsqueeze(0), self.cy.unsqueeze(0)), dim=0).T

        # Choose subset of data
        self.total_samples = len(self.u)
        self.choose_subset(self.subset, n=num_samples)

        # Get grid and time info
        self.x = torch.tensor(np.array(self.data["x"][:]))[...,::self.downsample, ::self.downsample].to(self.device)
        self.t = torch.tensor(np.array(self.data["t"][:])).to(self.device)

        # Get potentially useful variables from space and time
        self.tmin = self.t[0]
        self.tmax = self.t[-1]
        self.nt = len(self.t)
        self.dt = torch.Tensor([(self.tmax - self.tmin) / (self.nt-1)]*len(self.t)).unsqueeze(0)

        self.xmin = self.x[0, 0, 0]
        self.xmax = self.x[0, 0, -1]
        self.nx = len(self.x[0, 0])
        self.dx = (self.xmax - self.xmin)/ self.nx
        
        self.augmentation = augmentation
        self.shift = shift
        self.augmentation_ratio = augmentation_ratio

        f.close()

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
            #for idx in tqdm(range(self.u.shape[0])):
            for idx in tqdm(self.indexes):
                # Burgers
                if(self.nu[idx] != 0 and self.cx[idx] != 0 and self.cy[idx] != 0):
                    ratio = ((self.cx[idx] + self.cy[idx])**2)**(0.5)/self.nu[idx]
                    #ratios.append(ratio)
                    sentence = 'Burgers equation models a conservative system that can develop shock wave discontinuities.'
                    sentence += ' Burgers equation is a first order quasilinear hyperbolic partial differential equation.'
                    if(self.coeff):
                        sentence += ' In this case, the advection term has a coefficient of {} in the x direction, '
                        sentence += '{} in the y direction, and the diffusion term has a coefficient of {}.'.format(self.cx[idx],
                                                                                                          self.cy[idx], self.nu[idx])

                        if(self.qualitative):
                            cls = 'strongly' if(ratio > 100) else 'weakly'
                            sim = ' not ' if(ratio > 100) else ' '
                            sentence += ' This system is {} advection dominanted and does{}behave similarly to heat equation.'.format(cls, sim)
                            sentence += ' Ths predicted state should have shocks.' if(cls == 'strongly') else \
                                        ' The predicted state should look smoother than the inputs'
                # Advection
                elif(self.ax[idx] != 0 and self.ay[idx] != 0):
                    adv = ((self.ax[idx] + self.ay[idx])**2)**(0.5)
                    #advs.append(adv)
                    sentence = 'The Advection equation models bulk transport of a substance or quantity. It does not develop shocks.'
                    sentence += ' The Advection equation is a linear hyperbolic partial differential equation.'
                    if(self.coeff):
                        sentence += ' In this case, the advection term has a coefficient of {} in the x direction, '
                        sentence += '{} in the y direction.'.format(self.ax[idx], self.ay[idx])
    
                        if(self.qualitative):
                            cls = 'strongly' if(adv > 2) else 'weakly'
                            sentence += ' This system is {} advective.'.format(cls)
                            sentence += ' The predicted state should look like the input but shifted in space.'

                # Heat
                elif(self.nu[idx] != 0 and self.cx[idx] == 0 and self.cy[idx] == 0):
                    sentence = 'The Heat equation models how a quantity such as heat diffuses through a given region.'
                    sentence += ' The Heat equation is a linear parabolic partial differential equation.'
                    if(self.coeff):
                        sentence += ' In this case, the diffusion term has a coefficient of {}.'.format(self.nu[idx])

                        if(self.qualitative):
                            cls = 'strongly' if(self.nu[idx] > 0.01) else 'weakly'
                            sentence += ' This system is {} diffusive.'.format(cls)
                            sentence += ' The predicted state should look smoother than the inputs.'

                sentence += " This system has periodic boundary conditions."
                #sentence += " Give me an embedding that is useful for numerically predicting the target state."
                if(self.sentence):
                    #while(len(sentence) < 650): # Pad them to have same length
                    while(len(sentence) < 400): # Pad them to have same length
                        sentence += ' '
                    #while(len(sentence) < 400): # Pad them to have same length
                    #    sentence += ' '
                    if(len(sentence) > 650):
                    #if(len(sentence) > 400):
                        print(len(sentence))
                        raise
                    self.sentences.append(sentence)
                else:
                    self.sentence_embeddings.append(self.sentence_embedder.encode(sentence))
            print("Done.")

    def __len__(self):
        return len(self.indexes)#*self.u.shape[1]

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
        original_idx = idx
        #original_idx = idx//self.u.shape[1]
        #slice_idx = idx%self.u.shape[1]
        #print()
        #print(idx, original_idx, slice_idx)
        #print()
        idx = self.indexes[idx]
        #idx = self.indexes[idx//self.u.shape[1]]
        #print(idx)
        #raise
        u = self.u[idx]
        x = self.x
        t = self.t
        
        #variables = {}
        #variables['nu'] = self.nu[idx] 
        #variables['ax'] = self.ax[idx]
        #variables['ay'] = self.ay[idx]
        #variables['cx'] = self.cx[idx]
        #variables['cy'] = self.cy[idx]

        if self.mode == "train" and self.augmentation is not None:
            if self.augmentation_ratio > random.random(): # augment data w/ probability augmentation_ratio
                pde = self.get_PDE(variables)
                if not torch.is_tensor(u):
                    u = torch.tensor(u)
                u = self.augmentation(u, pde, self.shift)

        if(self.clip and not self.sentence):
            return_u = torch.cat((u.unsqueeze(-1), torch.zeros(u.shape[0], u.shape[1], u.shape[2], 3)), dim=-1)
            return return_u, \
                   x.permute(1,2,0), \
                   self.coeffs[original_idx], \
                   self.dt[0][0], \
                   self.sentence_embeddings[original_idx]

        elif(self.clip and self.sentence):
            return u, x.permute(1,2,0), variables, self.sentences[original_idx]
        else:
            return u, x.permute(1,2,0), variables

    def get_data(self):
        self.coeffs = self.coeffs[self.indexes]
        try:
            self.sentence_embeddings = torch.Tensor(self.sentence_embeddings)
        except AttributeError:
            self.sentence_embeddings = None
        #print()
        #print(type(self.sentence_embeddings))
        #print()
        #raise
        return self.u[self.indexes].unsqueeze(-1)
    

    def get_PDE(self, variables):
        if variables['ax'] != 0 and variables['ay'] != 0:
            return "advection"
        elif variables["cx"] != 0 and variables["cy"] != 0:
            return "burgers"
        elif variables["nu"] != 0:
            return "heat"
        else:
            raise ValueError("PDE not found")


    def choose_subset(
            self,
            chosen: str = 'heat,adv,burger',
            reverse: bool = False,
            n: int = None,
            ):
        """
        Choose subset of the dataset
        Args:
            chosen: str 
                stringof chosen PDEs and subset of PDE coefficients.
                DO NOT USE ANY SPACES!
                Example:
                    'heat,nu>0.5,adv,ax<0.4,burger,cx<0.3'

                Ranges:
                    nu:
                        - burgers: [7.5e-3, 1.5e-2]
                        - heat: [3e-3, 2e-2]
                    ax, ay: [0.1, 2.5]
                    cx, cy: [0.5, 1.0]

                    
            reverse: bool.
                if True, choose all PDEs except the specified ones
            n: int or None
                number of samples to use from the specified subset
            seed: int
                random seed when choosing n samples (for reproducibility)
        Returns:
            None
        """
        gs = chosen.split(',')

        if 'adv' in gs:
            adv = ((self.ax!=0) | (self.ay!=0)) & ((self.cx==0) & (self.cy==0)) & (self.nu==0)
        else:
            adv = torch.zeros(self.total_samples).bool()

        if 'burger' in gs:
            burger =((self.ax==0) & (self.ay==0)) & ((self.cx!=0) | (self.cy!=0)) & (self.nu!=0)
        else:
            burger = torch.zeros(self.total_samples).bool()

        if 'heat' in gs:
            heat = ((self.ax==0) & (self.ay==0)) & ((self.cx==0) & (self.cy==0)) & (self.nu!=0)
        else:
            heat = torch.zeros(self.total_samples).bool()

        if 'ns' in gs:
            ns = (self.visc != 0) & (self.amp != 0)
        else:
            ns = torch.zeros(self.total_samples).bool()

        for g in gs:
            if '>' in g:
                attr, val = g.split('>')
                if attr in ['ax', 'ay']:
                    adv = adv & (getattr(self, attr)>float(val))
                elif attr in ['cx', 'cy']:
                    burger = burger & (getattr(self, attr)>float(val))
                elif attr in ['nu']:
                    burger = burger & (getattr(self, attr)>float(val))
                    heat = heat & (getattr(self, attr)>float(val))
            elif '<' in g:
                attr, val = g.split('<')
                if attr in ['ax', 'ay']:
                    adv = adv & (getattr(self, attr)<float(val))
                elif attr in ['cx', 'cy']:
                    burger = burger & (getattr(self, attr)<float(val))
                elif attr in ['nu']:
                    burger = burger & (getattr(self, attr)<float(val))
                    heat = heat & (getattr(self, attr)<float(val))

        which = heat.to(self.device) | adv.to(self.device) | burger.to(self.device) | ns.to(self.device)
        if reverse:
            which = ~which

        self.indexes = torch.arange(self.total_samples, device=which.device)[which]

        if type(n) is int:
            if n > len(self.indexes):
                print(f"You want {n} samples but there are only {len(self.indexes)} available. Overriding {n} to {len(self.indexes)}")
                self.num_samples = len(self.indexes)
                n = len(self.indexes)

            self.indexes = self.indexes[np.random.choice(len(self.indexes), n, replace=False)]

        # Check number of equations
        eq_dict = {"heat": 0, "adv": 0, "burgers": 0}
        for idx in self.indexes:
            eq = self.get_eq(idx)
            eq_dict[eq] += 1

        print(eq_dict)


    def get_eq(self, idx):
        nu = self.nu[idx]
        cx = self.cx[idx]

        if nu == 0:
            return "adv"
        if cx == 0:
            return "heat"
        else:
            return "burgers"


