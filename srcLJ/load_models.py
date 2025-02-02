from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *


class LJFamily(torch.nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0, attr_pow=6,  rep_pow=12):
        super(LJFamily, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        self.attr_pow = attr_pow
        self.rep_pow = rep_pow 

    def LJ(self, r, sigma, epsilon):
        return 4 * epsilon * ((sigma/r)**self.rep_pow - (sigma/r)**self.attr_pow)

    def forward(self, x):
        return self.LJ(x, self.sigma, self.epsilon)

def get_model_list(data_str_list,system_list,nbr_list_device=['cpu'],device='cpu'):
    cutoff = 2.5
    mlp_parmas = {'n_gauss': int(cutoff//0.10), # cutoff//gaussian_width 
                'r_start': 0.0,
                'r_end': cutoff, 
                'n_width': 128,
                'n_layers': 3,
                'nonlinear': 'ELU'}
    lj_params = {'epsilon': 0.4, 
            'sigma': 0.9,
        "power": 10}

    NN = pairMLP(**mlp_parmas)
    pair = LJFamily(epsilon=2.0, sigma=0.9, rep_pow=6, attr_pow=3)  # ExcludedVolume(**lj_params)

    model_list = []
    for i, data_str in enumerate(data_str_list):

        pairNN = PairPotentials(system_list[i], NN,
                    cutoff=cutoff,
                    nbr_list_device=nbr_list_device
                    ).to(device)
        prior = PairPotentials(system_list[i], pair,
                        cutoff=2.5,
                    nbr_list_device=nbr_list_device
                        ).to(device)

        model = Stack({'pairnn': pairNN, 'pair': prior})
        model_list.append(model)
    
    return model_list, NN, pair