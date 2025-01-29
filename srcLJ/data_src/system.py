import ase
import torch
import numpy as np

class System(ase.Atoms):

    """Object that contains system information. Inherited from ase.Atoms
    
    Attributes:
        device (int or str): torch device "cpu" or an integer
        dim (int): dimension of the system (if n_dim < 3, the first n_dim columns are used for position calculation)
        props (dict{}): additional properties 
    """

    def __init__(
        self,
        *args,
        device,
        dim=3,
        props={},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.props = props
        self.device = device
        self.dim = dim
        
    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
            inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz
    
    def get_cell_len(self):
        return np.diag( self.get_cell() )

    def get_batch(self):
        batch = {
         'nxyz': torch.Tensor(self.get_nxyz()), 
         'num_atoms': torch.LongTensor([self.get_number_of_atoms()]),
         'energy': 0.0}
        
        return batch
        
    def set_temperature(self, T):
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 
        MaxwellBoltzmannDistribution(self, T)
        if self.dim < 3:
            vel =  self.get_velocities()
            vel[:, -1] = 0.0
            self.set_velocities(vel)
    
