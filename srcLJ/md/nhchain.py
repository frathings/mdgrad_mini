import torch
import numpy as np
from torch.autograd import grad

def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """
    Compute gradient of the scalar output with respect to inputs.    
    """
    assert inputs.requires_grad, "Inputs must have requires_grad=True"
    
    grad_val, = grad(
        output,
        inputs, 
        grad_outputs=output.data.new(output.shape).fill_(1),
        create_graph=create_graph,
        retain_graph=retain_graph
        )
    return grad_val

class NoseHooverChain(torch.nn.Module):    
    def __init__(self, potentials, system, T, num_chains=2, Q=1.0, adjoint=True
                ,topology_update_freq=1):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.device = system.device 

        # Set up masses and degrees of freedom.
        self.mass = torch.Tensor(system.get_masses()).to(self.device)
        self.T = T # Target temperature in eV
        self.N_dof = self.mass.shape[0] * system.dim
        self.target_ke = 0.5 * self.N_dof * T 
        
        # Thermostat (Nose-Hoover chain) parameters.
        self.num_chains = num_chains
        self.Q = np.array([Q, *[Q/len(self.system)]*(num_chains-1)]) #if your atoms are distributed, self.get_global_number_of_atoms.
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

    def update_T(self, new_T):
        self.T = new_T 
        
    def forward(self, t, state):
        """
        Compute the time derivatives for the state variables.

        Args:
            t (torch.Tensor): The current time (unused in this formulation).
            state (tuple): A tuple containing (velocities, positions, baths).

        Returns:
            tuple: Time derivatives (d(velocity)/dt, d(position)/dt, d(baths)/dt).
                   Note that d(position)/dt is simply the current velocity.
        """
        with torch.set_grad_enabled(True):            
            vel, pos, baths = state

            # Ensure positions have grad enabled if using adjoint sensitivity.
            if self.adjoint:
                pos.requires_grad = True
            
            N = self.N_dof
            
            # Compute momentum = mass * velocity.
            momentum = vel * self.mass[:, None]
            # Compute system kinetic energy.
            kinetic_energy = 0.5 * (momentum.pow(2) / self.mass[:, None]).sum()
            
            # Optionally update topology.
            self.update_topology(pos)           
            
            # Compute potential energy and the corresponding force.
            potential_energy = self.model(pos)
            # Sum potential energy along the last dimension and compute its gradient.
            force = -compute_grad(inputs=pos, output=potential_energy.sum(-1))

            # Compute coupling forces for the thermostat (using the first bath variable).
            coupled_forces = (baths[0] * momentum.reshape(-1) / self.Q[0]).reshape(-1, 3)

            # Compute acceleration (Newton's second law including thermostat coupling).
            acceleration = (force - coupled_forces) / self.mass[:, None]

            # Compute derivatives for thermostat variables.
            bath_deriv_0 = 2 * (kinetic_energy - self.T * self.N_dof * 0.5) - baths[0] * baths[1] / self.Q[1]
            bath_deriv_mid = (baths[:-2].pow(2) / self.Q[:-2] - self.T) - baths[2:] * baths[1:-1] / self.Q[2:]
            bath_deriv_last = baths[-2].pow(2) / self.Q[-2] - self.T

            # Concatenate thermostat derivatives.
            bath_deriv = torch.cat((bath_deriv_0[None], bath_deriv_mid, bath_deriv_last[None]))

        return (acceleration, vel, bath_deriv)

    def get_inital_states(self, wrap=True):
        print('nha')
        init_vel = self.system.get_velocities()
        init_pos = self.system.get_positions(wrap=wrap)
        # Initialize thermostat (bath) variables to zero.
        init_baths = [0.0] * self.num_chains

        # Convert all initial states to torch.Tensors on the correct device.
        states = [torch.Tensor(x).to(self.system.device) for x in [init_vel, init_pos, init_baths]]
        return states
