import torch
import numpy as np
from potential_src.pairMLP.potential_PairMLP import GaussianSmearing
from utils.topology import *
#from torchmd.system import check_system

class Observable(torch.nn.Module):
    def __init__(self, system):
        super(Observable, self).__init__()
        #check_system(system)
        self.device = system.device
        self.volume = system.get_volume()
        self.cell = torch.Tensor( system.get_cell()).diag().to(self.device)
       # self.natoms = system.get_number_of_atoms()
        self.natoms = len(system) # if distributed self.natoms = system.get_global_number_of_atoms()


def generate_vol_bins(start, end, nbins, dim):
    bins = torch.linspace(start, end, nbins + 1)
    
    # compute volume differential 
    if dim == 3:
        Vbins = 4 * np.pi /3*(bins[1:]**3 - bins[:-1]**3)
        V = (4/3)* np.pi * (end) ** 3
    elif dim == 2:
        Vbins = np.pi * (bins[1:]**2 - bins[:-1]**2)
        V = np.pi * (end) ** 2
        
    return V, torch.Tensor(Vbins), bins



class rdf(Observable):
    def __init__(self, system, nbins, r_range, index_tuple=None, width=None):
        super(rdf, self).__init__(system)
        PI = np.pi
        start = r_range[0]
        end = r_range[1]
        self.device = system.device
        V, vol_bins, bins = generate_vol_bins(start, end, nbins, dim=system.dim)
        self.V = V
        self.vol_bins = vol_bins.to(self.device)
        self.r_axis = np.linspace(start, end, nbins)
        self.device = system.device
        self.bins = bins
        self.smear = GaussianSmearing(
            start=start,
            stop=bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        ).to(self.device)

        self.nbins = nbins
        self.cutoff_boundary = end + 5e-1
        self.index_tuple = index_tuple
        
    def forward(self, xyz):
        nbr_list, pair_dis, _ = generate_nbr_list(xyz, 
                                               self.cutoff_boundary, 
                                               self.cell, 
                                               index_tuple=self.index_tuple, 
                                               get_dis=True)

        count = self.smear(pair_dis.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        count = count
        rdf =  count / (self.vol_bins / self.V )  

        return count, self.bins, rdf 

class vacf(Observable):
    def __init__(self, system, t_range):
        super(vacf, self).__init__(system)
        self.t_window = [i for i in range(1, t_range, 1)]

    def forward(self, vel):
        vacf = [(vel * vel).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in self.t_window]

        return torch.stack(vacf).reshape(-1)
