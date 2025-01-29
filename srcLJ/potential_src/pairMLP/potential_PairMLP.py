import torch
from torch import nn


def JS_rdf(g_obs, g):
    e0 = 1e-4
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js

nlr_dict =  {
    'ReLU': torch.nn.ReLU(), 
    'ELU': torch.nn.ELU(),
    'Tanh': torch.nn.Tanh(),
    'LeakyReLU': torch.nn.LeakyReLU(),
    'ReLU6': torch.nn.ReLU6(),
    'SELU': torch.nn.SELU(),
    'CELU': torch.nn.CELU(),
    'Tanhshrink': torch.nn.Tanhshrink()
}


def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss

class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    sample struct dictionary:

        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(distances,
                                   self.offsets,
                                   self.width,
                                   centered=self.centered)

        return result

class LennardJones(torch.nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0):
        super(LennardJones, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))

    def LJ(self, r, sigma, epsilon):
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    def forward(self, x):
        return self.LJ(x, self.sigma, self.epsilon)

class pairMLP(torch.nn.Module):
    def __init__(self, n_gauss, r_start, r_end, n_layers, n_width, nonlinear, res=False):
        super(pairMLP, self).__init__()
        

        nlr = nlr_dict[nonlinear]

        self.smear = GaussianSmearing(
            start=r_start,
            stop=r_end,
            n_gaussians=n_gauss,
            trainable=True
        )
        
        self.layers = nn.ModuleList(
            [
            nn.Linear(n_gauss, n_gauss),
            nlr,
            nn.Linear(n_gauss, n_width),
            nlr]
            )

        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_width, n_width))
            self.layers.append(nlr)

        self.layers.append(nn.Linear(n_width, n_gauss))  
        self.layers.append(nlr)  
        self.layers.append(nn.Linear(n_gauss, 1)) 
        self.res = res  # flag for residue connections 

        
    def forward(self, r):
        r = self.smear(r)
        for i in range(len(self.layers)):
            if self.res is False:
                r = self.layers[i](r)
            else:
                dr = self.layers[i](r)
                if dr.shape[-1] == r.shape[-1]:
                    r = r + dr 
                else:
                    r = dr 
        return r

def get_pair_potential(assignments, sys_params):

    cutoff = assignments['cutoff']
    mlp_parmas = {'n_gauss': int(cutoff//assignments['gaussian_width']), 
              'r_start': 0.0,
              'r_end': cutoff, 
              'n_width': assignments['n_width'],
              'n_layers': assignments['n_layers'],
              'nonlinear': assignments['nonlinear'],
              'res': False}

    lj_params = {'epsilon': assignments['epsilon'], 
         'sigma': assignments['sigma'],
       }

    net = pairMLP(**mlp_parmas)

    #net = pairTab(rc=10.0, device=sys_params['device'])

    prior = LennardJones(**lj_params)

    return net, prior



