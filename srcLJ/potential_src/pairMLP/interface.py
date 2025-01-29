import torch
from torch.nn import ModuleDict
from utils.topology import *

class Stack(torch.nn.Module):

    """Summary
    
    Attributes:
        models (TYPE): Description
    """
    
    def __init__(self, model_dict, mode='sum'):
        """Summary
        
        Args:
            model_dict (TYPE): Description
            mode (str, optional): Description
        """
        super().__init__()
        self.models = ModuleDict(model_dict)

    def _reset_topology(self, x):

        for i, key in enumerate(self.models.keys()):
            self.models[key]._reset_topology(x)
        
    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).sum().reshape(-1)
            else:
                new_result = self.models[key](x).sum().reshape(-1)
                result += new_result
        
        return result


class GeneralInteraction(torch.nn.Module):

    """Interface for energy calculator that requires dynamic update of interaction topology 
    
        example:
            Graph neural networks (GNNPotentials) need to update its topology given coordinates update during the simulation 
            Pair potentials (PairPotentials) need to search for interaction neighbor list 
    
    Attributes:
        cell (torch.Tensor): N x N tensor that specifies the basis of the simulation box 
        device (int or str): integer if for GPU; "cpu" for cpu  
        system (torchmd.system.System): Object to contain simulation information 
    """
    
    def __init__(self, system):
        """Summary
        
        Args:
            system (TYPE): Description
        """
        super(GeneralInteraction, self).__init__()
        self.system = system
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell.requires_grad = True 
        self.device = system.device


class PairPotentials(GeneralInteraction):

    """Summary
    
    Attributes:
        cutoff (TYPE): Description
        ex_pairs (TYPE): Description
        index_tuple (TYPE): Description
        model (TYPE): Description
    """
    
    def __init__(self, system, pair_model, cutoff=2.5, index_tuple=None, ex_pairs=None, nbr_list_device=None):
        """Summary
        
        Args:
            system (TYPE): Description
            pair_model (TYPE): Description
            cutoff (float, optional): Description
            index_tuple (None, optional): Description
            ex_pairs (None, optional): Description
        """
        super().__init__(system)

        if nbr_list_device == None:
            self.nbr_list_device = system.device
        else:
            self.nbr_list_device = nbr_list_device

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
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
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
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        #nbr_list, pair_dis, _ = self._reset_topology(xyz)

        # directly compute distances 
        # compute pair energy 

        pair_dis = compute_dis(xyz, self.nbr_list, self.offsets, self.cell)
        energy = self.model(pair_dis).sum()
        return energy

