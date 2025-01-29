import torch
from ase import units
from ase.geometry import wrap_positions

from md.solvers import odeint_adjoint, odeint

class Simulations():

    """Simulation object for handing runnindg MD and logging
    
    Attributes:
        device (str or int): int for GPU, "cpu" for cpu
        integrator (nn.module): function that updates force and velocity n
        keys (list): name of the state variables e.g. "velocities" and "positions "
        log (dict): save state vaiables in numpy arrays 
        solvemethod (str): integration method, current options are 4th order Runge-Kutta (rk4) and Verlet 
        system (torch.System): System object to contain state of molecular systems 
        wrap (bool): if True, wrap the coordinates based on system.cell 
    """
    
    def __init__(self,
                 system,
                  integrator,
                  wrap=True,
                  method="NH_verlet"):

        self.system = system 
        self.device = system.device
        self.integrator = integrator
        self.solvemethod = method
        self.wrap = wrap
        self.keys = self.integrator.state_keys
        self.initialize_log()

    def initialize_log(self):
        self.log = {}
        for key in self.keys:
            self.log[key] = []

    def update_log(self, trajs):
        for i, key in enumerate( self.keys ):
            if trajs[i][0].device != 'cpu':
                self.log[key].append(trajs[i][-1].detach().cpu().numpy()) 
            else:
                self.log[key].append(trajs[i][-1].detach().numpy()) 

    def update_states(self):
        if "positions" in self.log.keys():
            self.system.set_positions(self.log['positions'][-1])
        if "velocities" in self.log.keys():
            self.system.set_velocities(self.log['velocities'][-1])

    def get_check_point(self):

        if hasattr(self, 'log'):
            states = [torch.Tensor(self.log[key][-1]).to(self.device) for key in self.log]

            if self.wrap:
                wrapped_xyz = wrap_positions(self.log['positions'][-1], self.system.get_cell())
                states[1] = torch.Tensor(wrapped_xyz).to(self.device)

            return states 
        else:
            raise ValueError("No log available")
        
    def simulate(self, steps=1, dt=1.0 * units.fs, frequency=1):

        if self.log['positions'] == []:
            states = self.integrator.get_inital_states(self.wrap)
        else:
            states = self.get_check_point()

        sim_epochs = int(steps//frequency)
        t = torch.Tensor([dt * i for i in range(frequency)]).to(self.device)

        for epoch in range(sim_epochs):

            if self.integrator.adjoint:
                trajs = odeint_adjoint(self.integrator, states, t, method=self.solvemethod)
            else:
                for var in states:
                    var.requires_grad = True 
                trajs = odeint(self.integrator, tuple(states), t, method=self.solvemethod)
            self.update_log(trajs)
            self.update_states()

            states = self.get_check_point()

        return trajs
