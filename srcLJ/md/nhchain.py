import torch
import numpy as np
from torch.autograd import grad


def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """Compute gradient of the scalar output with respect to inputs.
    
    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 
    
    Returns:
        torch.Tensor: gradients with respect to each input component 
    """

    assert inputs.requires_grad
    
    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                   create_graph=create_graph, retain_graph=retain_graph)
    
    return gradspred

class NoseHooverChain(torch.nn.Module):

    """Equation of state for NVT integrator using Nose Hoover Chain 

    Nosé, S. A unified formulation of the constant temperature molecular dynamics methods. The Journal of Chemical Physics 81, 511–519 (1984).
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        mass (torch.Tensor): masses of each particle
        model (nn.module): energy functions that takes in coordinates 
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (torchmd.System): system object
        num_chains (int): number of chains 
        Q (float): Heat bath mass
        T (float): Temperature
        target_ke (float): target Kinetic energy 
    """
    
    def __init__(self, potentials, system, T, num_chains=2, Q=1.0, adjoint=True
                ,topology_update_freq=1):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.device = system.device # should just use system.device throughout
        self.mass = torch.Tensor(system.get_masses()).to(self.device)
        self.T = T # in energy unit(eV)
        self.N_dof = self.mass.shape[0] * system.dim
        self.target_ke = (0.5 * self.N_dof * T )
        
        self.num_chains = num_chains
        self.Q = np.array([Q,
                   *[Q/len(self.system)]*(num_chains-1)]) #if your atoms are distributed, self.get_global_number_of_atoms.
        self.Q = torch.Tensor(self.Q).to(self.device)
        self.dim = system.dim
        self.adjoint = adjoint
        self.state_keys = ['velocities', 'positions', 'baths']
        self.topology_update_freq = topology_update_freq
        self.update_count = 0

    def update_topology(self, q):

        if self.update_count % self.topology_update_freq == 0:
            self.model._reset_topology(q)
        self.update_count += 1


    def update_T(self, T):
        self.T = T 
        
    def forward(self, t, state):
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            p = v * self.mass[:, None]

            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            self.update_topology(q)           
            
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))

            #coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dpdt = f #- coupled_forces

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T

            dvdt = dpdt / self.mass[:, None]

        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))

    def get_inital_states(self, wrap=True):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap), 
                [0.0] * self.num_chains]

        states = [torch.Tensor(var).to(self.system.device) for var in states]
        return states

