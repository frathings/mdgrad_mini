import ase
import os
import sys

from data_src.system import System

import json
import os
import ase
import numpy as np
import sys
#from torchmd.potentials import LennardJones
from scipy import interpolate
from ase.visualize import view
from ase.lattice.cubic import FaceCenteredCubic, Diamond
#from torchmd.observable import generate_vol_bins
import os
import json


def load_config(config_file="../config/config.json"):
    """
    Load configuration parameters from a JSON file.
    """
    # Get the directory of the current script
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, config_file)
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist at {config_path}.")

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Example usage
config = load_config()

# Get data dictionaries
pair_data_dict = config.get("pair_data_dict", {})
exp_rdf_data_dict = config.get("exp_rdf_data_dict", {})

def get_unit_len(rho, mass, N_unitcell):
    Na = 6.02214086 * 10 ** 23  # Avogadro's number
    N = (rho * 10 ** 6 / mass) * Na  # Number of molecules in 1m^3 of water
    rho = N / (10 ** 30)  # Number density in 1 A^3
    L = (N_unitcell / rho) ** (1 / 3)
    return L

def visualize_system_with_ase_3d(data_tag, size):
    """
    Visualize the molecular system using ASE's 3D viewer.
    """
    params = exp_rdf_data_dict[data_tag]
    L = get_unit_len(params['rho'], params['mass'], params['N_unitcell'])
    atoms = eval(params['cell'])(
        symbol=params['element'], size=(size, size, size),
        latticeconstant=L, pbc=True
    )
    view(atoms)

def get_system(data_tag, device, size):
    print(f"Data tag: {data_tag}")
    params = exp_rdf_data_dict[data_tag]
    rho = params['rho']
    mass = params['mass']
    T = params['T']

    # Initialize states with ASE
    cell_module = eval(params['cell'])
    N_unitcell = params['N_unitcell']
    L = get_unit_len(rho, mass, N_unitcell)

    print(f"Lattice param: {L:.3f} Å")

    atoms = cell_module(
        symbol=params['element'], size=(size, size, size),
        latticeconstant=L, pbc=True
    )
    system = System(atoms, device=device)
    system.set_temperature(T * ase.units.kB)
    return system


def get_temp(T_start, T_equil, n_epochs, i, anneal_rate):
    return (T_start - T_equil) * np.exp( - i * (1/n_epochs) * anneal_rate) + T_equil
