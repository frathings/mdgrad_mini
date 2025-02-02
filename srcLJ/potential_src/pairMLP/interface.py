import torch
from torch.nn import ModuleDict
from utils.topology import *

class Stack(torch.nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()
        self.models = ModuleDict(model_dict)

    def _reset_topology(self, x):
        for i, key in enumerate(self.models.keys()):
            self.models[key]._reset_topology(x)
        
    def forward(self, x):
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).sum().reshape(-1)
            else:
                new_result = self.models[key](x).sum().reshape(-1)
                result += new_result
        return result


class PairPotentials(torch.nn.Module):
    def __init__(self, system, pair_model, cutoff=2.5, index_tuple=None, ex_pairs=None, nbr_list_device=None):
        super().__init__()

        if nbr_list_device == None:
            self.nbr_list_device = system.device
        else:
            self.nbr_list_device = nbr_list_device
        self.system = system
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell.requires_grad = True 
        self.device = system.device

        self.model = pair_model
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs

        nbr_list, offsets = generate_nbr_list(
                                       torch.Tensor(
                                            system.get_positions()
                                                ).to(self.nbr_list_device), 
                                       self.cutoff, 
                                       self.cell.to(self.nbr_list_device), 
                                       index_tuple=self.index_tuple, 
                                       ex_pairs=self.ex_pairs)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(system.device)


    def _reset_topology(self, xyz):
        nbr_list, pair_dis, offsets = generate_nbr_list(xyz.to(self.nbr_list_device), 
                                               self.cutoff, 
                                               self.cell.to(self.nbr_list_device), 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(xyz.device)

        return nbr_list, pair_dis, offsets

    def forward(self, xyz):
        pair_dis = compute_dis(xyz, self.nbr_list, self.offsets, self.cell)
        energy = self.model(pair_dis).sum()
        return energy

