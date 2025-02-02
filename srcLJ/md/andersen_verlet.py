import torch
import numpy as np
from torch.autograd import grad

def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """
    Compute the gradient of the (scalar) output with respect to inputs.
    """
    grad_val, = grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph
    )
    return grad_val

class AndersenODE(torch.nn.Module):
    """
    A continuous approximation to the Andersen thermostat.
    
    Here we approximate the effect of Andersen collisions (which are discrete)
    by adding a friction (relaxation) term. This leads to a continuous ODE:
    
        d(pos)/dt = v
        d(v)/dt   = (force/m) - nu * v
    
    This integrator is intended to be used with your ODE solver (which uses the Verlet update)
    and therefore must provide a state with exactly two elements.
    """
    def __init__(self, potentials, system, T, dt, nu, adjoint=True):
        """
        Args:
            potentials: Callable that returns the potential energy given positions.
            system: An object with methods get_positions(), get_velocities(), get_masses(),
                    and attributes device and dim.
            T: Target temperature (for reference).
            dt: Time step (stored but not used directly in the derivative).
            nu: Effective collision (relaxation) rate.
        """
        super().__init__()
        self.model = potentials 
        self.system = system
        self.device = system.device
        
        self.T = T          # Target temperature (for reference)
        self.dt = dt        # Stored time step (for consistency)
        self.nu = nu        # Effective collision frequency (used as friction)
        
        # Get per-particle masses.
        self.mass = torch.Tensor(system.get_masses()).to(self.device)  # shape: (N,)
        self.dim = system.dim
        self.N = self.mass.shape[0]  # Number of particles.
        
        # IMPORTANT: Define state_keys as a two-element list.
        self.state_keys = ['positions', 'velocities']
        self.adjoint = adjoint

    def forward(self, t, state):
        """
        Compute the deterministic time derivatives of the state.
        
        Args:
            t (torch.Tensor): Current time (unused here).
            state (tuple): (positions, velocities).
            
        Returns:
            tuple: (dvel/dt, dpos/dt), where:
                dpos/dt = velocities,
                dvel/dt = (force/m).
        """
        with torch.set_grad_enabled(True):
            if len(state) > 2:
                state = state[:2]
            pos, vel = state
            
            if self.adjoint:
                pos = pos.clone().detach().requires_grad_(True)
            
            # Compute potential energy and forces
            potential_energy = self.model(pos)
            pot_sum = potential_energy.sum() if potential_energy.ndim > 0 else potential_energy
            force = -compute_grad(pos, pot_sum)
            acc = force / self.mass.unsqueeze(1)  # acceleration = F/m
            
            dvel = acc  # Remove stochastic and friction terms!
            dpos = vel
            
        return dvel, dpos

    def get_inital_states(self, wrap=True):
        print('andersen')
        init_vel = self.system.get_velocities()
        init_pos = self.system.get_positions(wrap=wrap)
        # Convert all initial states to torch.Tensors on the correct device.
        states = [torch.Tensor(x).to(self.system.device) for x in [init_vel, init_pos]]
        return states
