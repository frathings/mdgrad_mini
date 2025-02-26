import os
import json
import numpy as np

import matplotlib.pyplot as plt


import questionary
from rich import print, box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.visualize import *
from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *


############################
#   Parameters definition
############################

# Configuration dictionary with all parameters organized by category
CONFIG = {
    # System parameters for different materials/conditions
    "systems": {
        'lj_0.3_1.2': {
            'rho': 0.3,
            'T': 1.2,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones(sigma=1, epsilon=1),
            "cutoff": 2.5,
            "dim": 3,
            "dt": 0.01
        }
        # Add more system configurations as needed
    },
    
    # Simulation parameters
    "simulation": {
        "n_epochs": 1000,
        "n_sim": 200,
        "size": 4,
        "t_range": 50,
        "device": "cpu",
        "nbins": 100,
        "tau": 60,
        "skip": 1,
        "topology_update_freq": 1
    },
    
    # MD parameters
    "md": {
        "NHC": {  # Nose-Hoover Chain parameters
            "Q": 50.0,
            "num_chains": 5,
            "adjoint": True
        },
        "timestep": {
            "dt": 0.01,
            "frequency": 100
        }
    },
    
    # Observable parameters
    "observables": {
        "rdf": {
            "nbins": 100,
            "r_range": {
                "start": 0.1,
                "end": 3.3
            }
        },
        "vacf": {
            "skip": 25
        }
    }
}

# Global Console Instance
console = Console()

############################
#   Helper functions
############################

def display_data(data, title="Simulation Data"):
    """
    Displays simulation data in a formatted table with organized categories.
    
    Args:
        data (dict): Dictionary of parameters to display
        title (str): Title for the table
    """
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Category", style="blue", justify="left")
    table.add_column("Parameter", style="cyan", justify="left")
    table.add_column("Value", style="magenta", justify="right")
    
    # Define parameter categories and order
    categories = {
        "System Properties": ["rho", "T", "element", "mass", "cutoff", "dim"],
        "Cell Structure": ["N_unitcell", "cell"],
        "Potential": ["target_pot", "sigma", "epsilon"],
        "RDF Settings": ["r_range", "nbins"],
        "Simulation": ["n_epochs", "n_sim", "size", "device", "t_range", "tau", "skip", "topology_update_freq"],
        "MD Parameters": ["dt", "frequency", "Q", "num_chains", "adjoint"],
        "Other": []  # Catch-all for other parameters
    }
    
    # Process nested dictionaries (like r_range)
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_data[f"{key}.{subkey}"] = subvalue
        else:
            flat_data[key] = value
    
    # Track which parameters have been displayed
    displayed_params = set()
    
    # Display parameters by category
    for category, params in categories.items():
        shown_category = False
        for param in params:
            # Check for exact match or prefix match (for nested params)
            matches = [k for k in flat_data.keys() if k == param or k.startswith(f"{param}.")]
            for key in matches:
                if not shown_category:
                    category_name = category
                    shown_category = True
                else:
                    category_name = ""      
                value = flat_data[key]
                # Format special values
                if isinstance(value, type):
                    value = value.__name__
                elif key.endswith("_pot") and hasattr(value, "__class__"):
                    value = value.__class__.__name__
                table.add_row(category_name, key, str(value))
                displayed_params.add(key)
    
    # Display any remaining parameters
    remaining = [k for k in flat_data.keys() if k not in displayed_params]
    if remaining:
        shown_category = False
        for key in sorted(remaining):
            if not shown_category:
                category_name = "Other"
                shown_category = True
            else:
                category_name = ""
            table.add_row(category_name, key, str(flat_data[key]))
    console.print(table)

def get_user_input(default_system='lj_0.3_1.2'):
    """Prompts the user to input simulation parameters."""
    system_data = CONFIG["systems"][default_system].copy()
    rdf_r_range = CONFIG["observables"]["rdf"]["r_range"].copy()
    
    # Get system parameters
    for param in ['rho', 'T']:
        system_data[param] = float(questionary.text(
            f"Enter {param}:", 
            default=str(system_data[param])
        ).ask())
    
    # Get RDF r_range parameters
    for param in ['start', 'end']:
        rdf_r_range[param] = float(questionary.text(
            f"Enter RDF r_range.{param}:", 
            default=str(rdf_r_range[param])
        ).ask())
    
    system_data['element'] = questionary.text(
        "Enter the chemical element:", 
        default=system_data['element']
    ).ask()
    
    # Return both updated system data and updated RDF range
    return {
        'system': system_data,
        'observables': {
            'rdf': {
                'r_range': rdf_r_range
            }
        }
    }

def update_config(new_params, system_key='lj_0.3_1.2'):
    """Updates the configuration with new parameters."""
    if 'system' in new_params:
        CONFIG["systems"][system_key].update(new_params['system'])
    if 'simulation' in new_params:
        CONFIG["simulation"].update(new_params['simulation'])
    if 'md' in new_params:
        for md_key, md_params in new_params['md'].items():
            if md_key in CONFIG["md"]:
                CONFIG["md"][md_key].update(md_params)
    if 'observables' in new_params:
        for obs_key, obs_params in new_params['observables'].items():
            if obs_key in CONFIG["observables"]:
                CONFIG["observables"][obs_key].update(obs_params)

def get_simulation_params():
    """Returns a dictionary of simulation parameters."""
    sim_params = CONFIG["simulation"].copy()
    # Add any system-specific parameters needed for the simulation
    system_key = 'lj_0.3_1.2'  # Default system, could be passed as parameter
    sim_params["cutoff"] = CONFIG["systems"][system_key]["cutoff"]
    sim_params["rdf_start"] = CONFIG["observables"]["rdf"]["r_range"]["start"]
    return sim_params

############################
#   Plotting functions
############################

def plot_vacf(vacf_sim, vacf_target, fn, path, dt=None, save_data=False):
    """Plot velocity autocorrelation function."""
    if dt is None:
        dt = CONFIG["md"]["timestep"]["dt"]  # Use default from config
    
    t_range = np.linspace(0.0, vacf_sim.shape[0] * dt, vacf_sim.shape[0])

    plt.figure(figsize=(8, 5))
    plt.plot(t_range, vacf_sim, label='Simulation', linewidth=4, alpha=0.6)

    if vacf_target is not None:
        plt.plot(t_range, vacf_target, label='Target', linewidth=2, linestyle='--', color='black')

    plt.xlabel("Time (fs)")
    plt.ylabel("VACF")
    plt.title("Velocity Auto-Correlation Function (VACF)")
    plt.legend()
    plt.grid()
    # plt.show()

    if save_data:
        np.savetxt(f"{path}/vacf_{fn}.txt", np.stack((t_range, vacf_sim), axis=1), delimiter=',')
        np.savetxt(f"{path}/vacf_{fn}_target.txt", np.stack((t_range, vacf_target), axis=1), delimiter=',')

    plt.savefig(f"{path}/vacf_{fn}.pdf", bbox_inches='tight')
    plt.close()

def plot_rdf( g_sim, rdf_target, fn, path, start, nbins, save_data=False, end=3.3):

    bins = np.linspace(start, end, nbins)

    plt.plot(bins, g_sim , label='simulation', linewidth=4, alpha=0.6)
    plt.plot(bins, rdf_target , label='target', linewidth=2,linestyle='--', c='black')
    
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")

    if save_data:
        np.savetxt(path + '/rdf_{}.txt'.format(fn), np.stack((bins, g_sim)), delimiter=',' )
        np.savetxt(path + '/rdf_{}_target.txt'.format(fn), np.stack((bins, rdf_target)), delimiter=',' )

    plt.savefig(path + '/rdf_{}.pdf'.format(fn), bbox_inches='tight')
    plt.close()

def plot_pair(fn, path, prior, device, end=2.5, target_pot=None, model=None): 
    if target_pot is None:
        target_pot = LennardJones(sigma=1, epsilon=1)
    else:
        target_pot = target_pot.to("cpu")

    x = torch.linspace(0.1, end, 250)[:, None].to(device)
    
    #u_fit = (model(x) + prior(x)).detach().cpu().numpy()
    u_fit = (prior(x)).detach().cpu().numpy()
    u_fit = u_fit - u_fit[-1] 

    u_target = target_pot(x.detach().cpu()).squeeze()

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)
    
    plt.plot( x.detach().cpu().numpy(), 
              u_target.detach().cpu().numpy(),
               label='truth', 
               linewidth=2,linestyle='--', c='black')

    plt.ylim(-2, 4.0)
    plt.legend()      
    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

    return u_fit

############################
#    Build Simulation
############################

def get_system(system_key, device, size=None):
    """Build a simulation system based on configuration parameters."""
    if size is None:
        size = CONFIG["simulation"]["size"]
        
    system_config = CONFIG["systems"][system_key]
    rho = system_config['rho']
    T = system_config['T']
    dim = system_config.get("dim", 3)
    
    if dim == 3:
        # initialize states with ASE 
        cell_module = system_config['cell']
        N_unitcell = system_config['N_unitcell']
        
        def get_unit_len(rho, N_unitcell):
            L = (N_unitcell / rho) ** (1/3)
            return L
            
        L = get_unit_len(rho, N_unitcell)
        print("lattice param:", L)
        
        atoms = cell_module(
            symbol=system_config['element'],
            size=(size, size, size),
            latticeconstant=L,
            pbc=True
        )
        
        system = System(atoms, device=device)
        system.set_temperature(T)
        
    return system 

def get_target_obs(system, system_key, n_sim=None, rdf_range=None, nbins=None, t_range=None, dt=None, skip=None):
    """Generate target observables from MD simulation."""
    print(f"Simulating {system_key}")
    
    # Get parameters from config
    system_config = CONFIG["systems"][system_key]
    sim_config = CONFIG["simulation"]
    md_config = CONFIG["md"]
    obs_config = CONFIG["observables"]
    rdf_config = obs_config["rdf"]
    
    # Use provided parameters or defaults from config
    if n_sim is None:
        n_sim = sim_config["n_sim"]
    if nbins is None:
        nbins = rdf_config["nbins"]
    if t_range is None:
        t_range = sim_config["t_range"]
    if dt is None:
        dt = md_config["timestep"]["dt"]
    if skip is None:
        skip = obs_config["vacf"]["skip"]
    if rdf_range is None:
        rdf_range = (rdf_config["r_range"]["start"], rdf_config["r_range"]["end"])
    
    device = system.device 
    
    # simulation setup
    target_pot = system_config['target_pot']  # take target potential
    T = system_config['T']  # take target temperature
    cutoff = system_config['cutoff']
    
    pot = PairPotentials(system, target_pot, cutoff=cutoff, nbr_list_device=device).to(device)
    
    diffeq = NoseHooverChain(
        pot, 
        system,
        Q=md_config["NHC"]["Q"], 
        T=T,
        num_chains=md_config["NHC"]["num_chains"], 
        adjoint=md_config["NHC"]["adjoint"],
        topology_update_freq=sim_config["topology_update_freq"]
    ).to(system.device)
    
    # define simulator 
    sim = Simulations(system, diffeq)
    
    # define objects for the observables
    rdf_obs = rdf(system, nbins=nbins, r_range=rdf_range)
    vacf_obs = vacf(system, t_range=t_range) 
    
    all_vacf_sim = []
    
    # Run MD Simulations & Extract Target Data
    freq = md_config["timestep"]["frequency"]
    for i in range(n_sim):
        v_t, q_t, pv_t = sim.simulate(freq, dt=dt, frequency=freq)
        if i >= skip:
            vacf_sim = vacf_obs(v_t).detach().cpu().numpy()
            all_vacf_sim.append(vacf_sim)
            
    # loop over to compute observables 
    trajs = torch.Tensor(np.stack(sim.log['positions'])).to(system.device).detach()
    all_g_sim = []
    
    for i in range(len(trajs)):
        if i >= skip:
            _, _, g_sim = rdf_obs(trajs[[i]])
            all_g_sim.append(g_sim.detach().cpu().numpy())
            
    all_g_sim = np.array(all_g_sim).mean(0)
    all_vacf_sim = np.array(all_vacf_sim).mean(0)

    return all_g_sim, all_vacf_sim

def get_observer(system, system_key, nbins=None, t_range=None, rdf_start=None):
    """Initialize and return observables for the system."""
    system_config = CONFIG["systems"][system_key]
    sim_config = CONFIG["simulation"]
    obs_config = CONFIG["observables"]
    rdf_config = obs_config["rdf"]
    
    # Use provided parameters or defaults
    if nbins is None:
        nbins = rdf_config["nbins"]
    if t_range is None:
        t_range = sim_config["t_range"]
    if rdf_start is None:
        rdf_start = rdf_config["r_range"]["start"]
    
    # get dt 
    dt = system_config.get('dt', CONFIG["md"]["timestep"]["dt"])
    rdf_end = rdf_config["r_range"].get("end", system_config["cutoff"])
    
    xnew = np.linspace(rdf_start, rdf_end, nbins)
    
    # initialize observable function 
    obs = rdf(system, nbins, (rdf_start, rdf_end))
    vacf_obs = vacf(system, t_range=t_range) 
    
    # get experimental rdf 
    dim = system_config.get("dim", 3) 
    rdf_data_path = system_config.get("fn", None)
    
    # generate simulated data 
    if not rdf_data_path:
        rdf_data, vacf_target = get_target_obs(
            system, 
            system_key, 
            n_sim=sim_config["n_sim"], 
            rdf_range=(rdf_start, rdf_end), 
            nbins=nbins, 
            t_range=t_range, 
            skip=obs_config["vacf"]["skip"], 
            dt=dt
        )
        vacf_target = torch.Tensor(vacf_target).to(system.device)
        rdf_data = np.vstack((np.linspace(rdf_start, rdf_end, nbins), rdf_data))
    else:
        # experimental rdfs
        rdf_data = np.loadtxt(rdf_data_path, delimiter=',')
        vacf_target = None

    _, rdf_target = get_exp_rdf(rdf_data, nbins, (rdf_start, rdf_end), obs.device, dim=dim)
    
    # get model potential and simulate 
    return xnew, rdf_target, obs, vacf_target, vacf_obs

def get_sim(system, model, system_key, topology_update_freq=None):
    """Initialize and return a simulation object."""
    system_config = CONFIG["systems"][system_key]
    md_config = CONFIG["md"]["NHC"]
    sim_config = CONFIG["simulation"]
    
    if topology_update_freq is None:
        topology_update_freq = sim_config["topology_update_freq"]
    
    T = system_config['T']
    
    diffeq = NoseHooverChain(
        model, 
        system,
        Q=md_config["Q"], 
        T=T,
        num_chains=md_config["num_chains"], 
        adjoint=md_config["adjoint"],
        topology_update_freq=topology_update_freq
    ).to(system.device)
    
    # define simulator
    sim = Simulations(system, diffeq)

    return sim

class LJFamily(nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0, attr_pow=6, rep_pow=12):
        super(LJFamily, self).__init__()
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32, requires_grad=True))
        self.epsilon = nn.Parameter(torch.tensor([epsilon], dtype=torch.float32, requires_grad=True))
        self.attr_pow = attr_pow
        self.rep_pow = rep_pow

    def LJ(self, r):
        return 4 * self.epsilon * ((self.sigma / r)**self.rep_pow - (self.sigma / r)**self.attr_pow)

    def forward(self, x):
        return self.LJ(x)
    

def create_models(system_list, data_str_list, val_str_list, device, cutoff=2.5, nbr_list_device=None):
    """
    Create models based on user selection (NN only, Prior only, or Both).
    
    Args:
        system_list: List of molecular systems
        data_str_list: List of training system identifiers
        val_str_list: List of validation system identifiers
        device: Computation device (CPU/GPU)
        cutoff: Cutoff distance for pair potentials
        nbr_list_device: Device for neighbor list calculation
        
    Returns:
        List of model objects
    """
    # Define parameters for neural network and LJ potential
    mlp_params = {
        'n_gauss': int(cutoff//0.10),  # cutoff//gaussian_width 
        'r_start': 0.0,
        'r_end': cutoff, 
        'n_width': 128,
        'n_layers': 3,
        'nonlinear': 'ELU'
    }
    
    lj_params = {
        'epsilon': 1.0, 
        'sigma': 1.0,
        'rep_pow': 12,
        'attr_pow': 6
    }
    
    # Prompt user to select model type
    model_type = questionary.select(
        "Select model type:",
        choices=[
            "Neural Network Only",
            "Lennard-Jones Prior Only",
            "Combined (NN + LJ Prior)"
        ]
    ).ask()
    
    # Initialize models
    NN = None
    pair = None
    if model_type == "Combined (NN + LJ Prior)":
        # Allow user to modify parameters
        modify_params = questionary.confirm("Do you want to modify model parameters?", default=False).ask()
        
        if modify_params:
            # Allow user to modify MLP parameters
            console.print(Panel("Neural Network (MLP) Parameters", style="cyan"))
            for param in ['n_width', 'n_layers']:
                mlp_params[param] = int(questionary.text(
                    f"Enter {param}:", 
                    default=str(mlp_params[param])
                ).ask())
            
            # Allow user to modify LJ parameters
            console.print(Panel("Lennard-Jones Parameters", style="green"))
            for param in ['epsilon', 'sigma', 'rep_pow', 'attr_pow']:
                lj_params[param] = float(questionary.text(
                    f"Enter {param}:", 
                    default=str(lj_params[param])
                ).ask())
        
    if model_type == "Lennard-Jones Prior Only":
        # Allow user to modify parameters
        modify_params = questionary.confirm("Do you want to modify model parameters?", default=False).ask()
        
        if modify_params:
            # Allow user to modify LJ parameters
            console.print(Panel("Lennard-Jones Parameters", style="green"))
            for param in ['epsilon', 'sigma', 'rep_pow', 'attr_pow']:
                lj_params[param] = float(questionary.text(
                    f"Enter {param}:", 
                    default=str(lj_params[param])
                ).ask())
        

    if model_type in ["Neural Network Only", "Combined (NN + LJ Prior)"]:
        NN = pairMLP(**mlp_params)
        console.print(f"[bold cyan]Neural Network initialized with {mlp_params['n_layers']} layers, width {mlp_params['n_width']}[/bold cyan]")
    
    if model_type in ["Lennard-Jones Prior Only", "Combined (NN + LJ Prior)"]:
        pair = LJFamily(
            epsilon=lj_params['epsilon'], 
            sigma=lj_params['sigma'], 
            rep_pow=lj_params['rep_pow'], 
            attr_pow=lj_params['attr_pow']
        )
        console.print(f"[bold green]LJ potential initialized with ε={lj_params['epsilon']}, σ={lj_params['sigma']}[/bold green]")
    
    # Create models for each system
    model_list = []
    console.print("[bold]Creating models for each system...[/bold]")
    
    for i, data_str in enumerate(data_str_list + val_str_list):
        if model_type == "Neural Network Only":
            pairNN = PairPotentials(
                system_list[i], 
                NN,
                cutoff=cutoff,
                nbr_list_device=nbr_list_device
            ).to(device)
            model = Stack({'pairnn': pairNN})
            
        elif model_type == "Lennard-Jones Prior Only":
            prior = PairPotentials(
                system_list[i], 
                pair,
                cutoff=cutoff,
                nbr_list_device=nbr_list_device
            ).to(device)
            model = Stack({'pair': prior})
            
        else:  # Combined
            pairNN = PairPotentials(
                system_list[i], 
                NN,
                cutoff=cutoff,
                nbr_list_device=nbr_list_device
            ).to(device)
            
            prior = PairPotentials(
                system_list[i], 
                pair,
                cutoff=cutoff,
                nbr_list_device=nbr_list_device
            ).to(device)
            
            model = Stack({'pairnn': pairNN, 'pair': prior})
        
        model_list.append(model)
        console.print(f"[green]✓[/green] Model created for system {data_str}")
    
    return model_list

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"{name}: {list(param.shape)} ({param_count} parameters)")
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params
def setup_optimizer_and_scheduler(models):
    """
    Sets up optimizer and learning rate scheduler based on model type.
    
    Args:
        models: List of model objects
        
    Returns:
        tuple: (optimizer, scheduler, model_type, loss_log, obs_log)
    """
    console.print("\n[bold]Setting up optimizer and learning rate scheduler...[/bold]")
    
    # Use first model as reference
    model = models[0]
    model_type = ""
    nn_model = None
    lj_model = None
    
    # Debug model structure
    console.print(f"Model type: {type(model).__name__}")
    
    # Check if model has models attribute (Stack objects use this)
    if hasattr(model, 'models'):
        console.print("Found 'models' attribute")
        potential_keys = list(model.models.keys())
        console.print(f"Found models: {potential_keys}")
        
        # Examine each model to determine its type
        for key in potential_keys:
            pair_pot = model.models[key]
            console.print(f"Examining model '{key}' of type {type(pair_pot).__name__}")
            
            # Check if this is a PairPotentials object
            if type(pair_pot).__name__ == 'PairPotentials':
                # PairPotentials should have a 'model' attribute
                if hasattr(pair_pot, 'model'):
                    pot_model = pair_pot.model
                    console.print(f"Found potential model of type: {type(pot_model).__name__}")
                    
                    # Identify the model type
                    if type(pot_model).__name__ == 'pairMLP':
                        nn_model = pot_model
                        if not model_type:
                            model_type = "Neural Network (NN)"
                        elif model_type == "Lennard-Jones (LJ)":
                            model_type = "Combined (NN + LJ)"
                    elif hasattr(pot_model, 'sigma'):
                        lj_model = pot_model
                        if not model_type:
                            model_type = "Lennard-Jones (LJ)"
                        elif model_type == "Neural Network (NN)":
                            model_type = "Combined (NN + LJ)"
    
    # If we couldn't determine the model type from the structure, try parameters
    if not model_type:
        console.print("[yellow]Trying to determine model type from parameters...[/yellow]")
        for name, param in model.named_parameters():
            console.print(f"Parameter: {name}, Shape: {param.shape}")
            if 'sigma' in name:
                if 'model.models.pair.model' in name:
                    lj_model = model.models['pair'].model
                    model_type = "Lennard-Jones (LJ)"
                    break
            if 'weight' in name and not 'sigma' in name:
                model_type = "Neural Network (NN)"
                if 'model.models.pair.model' in name:
                    nn_model = model.models['pair'].model
    
    # Print detected models and parameters
    if nn_model:
        console.print(f"[cyan]Found Neural Network model: {type(nn_model).__name__}[/cyan]")
    if lj_model:
        console.print(f"[green]Found LJ model: {type(lj_model).__name__}[/green]")
        if hasattr(lj_model, 'sigma'):
            try:
                console.print(f"[green]LJ parameters: sigma={lj_model.sigma.item():.3f}, epsilon={lj_model.epsilon.item():.3f}[/green]")
            except:
                console.print("[yellow]Could not display LJ parameters[/yellow]")
    
    console.print(f"Detected model type: [bold cyan]{model_type}[/bold cyan]")
    
    # Initialize optimizer based on model type
    optimizer = None
    
    if model_type == "Combined (NN + LJ)" and nn_model and lj_model:
        # Set up optimizer for both NN and LJ parameters
        optimizer = torch.optim.Adam([
            {'params': nn_model.parameters(), 'lr': 0.001},
            {'params': lj_model.sigma, 'lr': 0.01},
            {'params': lj_model.epsilon, 'lr': 0.05}
        ])
        console.print("Optimizing: [bold]Neural Network + Lennard-Jones parameters[/bold]")
        
    elif model_type == "Neural Network (NN)" and nn_model:
        # Set up optimizer for NN parameters only
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
        console.print("Optimizing: [bold]Neural Network parameters[/bold]")
        
    elif model_type == "Lennard-Jones (LJ)" and lj_model:
        # Set up optimizer for LJ parameters only
        optimizer = torch.optim.Adam([
            {'params': lj_model.sigma, 'lr': 0.01},
            {'params': lj_model.epsilon, 'lr': 0.05}
        ])
        console.print("Optimizing: [bold]Lennard-Jones parameters (sigma, epsilon)[/bold]")
    
    # Set up learning rate scheduler
    scheduler = None
    if optimizer:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min', 
            min_lr=1e-6, 
            verbose=True, 
            factor=0.5, 
            patience=20,
            threshold=5e-5
        )
        console.print(f"Learning rate scheduler: [bold]ReduceLROnPlateau[/bold] (factor=0.5, patience=20)")
    else:
        console.print("[bold red]Error: Could not determine model type to set up optimizer[/bold red]")
    
    # Initialize logs
    loss_log = []
    obs_log = dict()
    
    if optimizer and scheduler:
        console.print("[green]✓[/green] Optimizer and scheduler set up successfully")
    
    return optimizer, scheduler, model_type, loss_log, obs_log


############################
# Training
############################
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn

def train_model(models, sim_list, rdf_obs_list, vacf_obs_list, rdf_target_list, 
               vacf_target_list, rdf_bins_list, optimizer, scheduler, device,
               n_epochs=1000, model_path="./results", rdf_weight=1.0, vacf_weight=0.1,
               train_vacf="True", cutoff=2.5, plot_interval=10, target_pot=None):
    """
    Train the model using RDF and VACF targets.
    
    Args:
        models: List of model objects
        sim_list: List of simulation objects
        rdf_obs_list: List of RDF observer objects
        vacf_obs_list: List of VACF observer objects
        rdf_target_list: List of RDF target data
        vacf_target_list: List of VACF target data
        rdf_bins_list: List of RDF bin arrays
        optimizer: Optimizer object
        scheduler: Learning rate scheduler
        device: Computation device
        n_epochs: Number of epochs to train
        model_path: Path to save results
        rdf_weight: Weight for RDF loss
        vacf_weight: Weight for VACF loss
        train_vacf: Whether to use VACF in training ("True" or "False")
        cutoff: Cutoff distance
        plot_interval: Interval for plotting and saving results
        target_pot: Target potential (for comparison)
        
    Returns:
        tuple: (loss_log, obs_log)
    """
    # Create model path if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Get system info
    data_str_list = [f"system_{i}" for i, sim in enumerate(sim_list)]
    val_str_list = []  # Define your validation list if needed
    
    # Initialize logs
    loss_log = []
    obs_log = {data_str: {'rdf': [], 'vacf': []} for data_str in data_str_list + val_str_list}
    
    # Set parameters for simulation
    tau = 60  # Steps per simulation
    t_range = 50  # Time range for VACF
    skip = 5  # Skip frames for RDF calculation
    dt = 0.005  # Timestep
    nbins = 100  # Number of bins for RDF
    
    # Main training loop
    console.print(f"\n[bold]Starting training for {n_epochs} epochs...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[bold green]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Create a task for training
        train_task = progress.add_task("[bold]Training...", total=n_epochs)
        
        for i in range(n_epochs):
            # Update task description
            progress.update(train_task, 
                           description=f"[bold]Training... Epoch {i+1}/{n_epochs}",
                           advance=0)
            
            loss_rdf = torch.Tensor([0.0]).to(device)
            loss_vacf = torch.Tensor([0.0]).to(device)
            
            # Train on each system
            n_train = len(data_str_list)
            for j, sim in enumerate(sim_list[:n_train]):
                data_str = (data_str_list + val_str_list)[j]
                
                # Simulate
                v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=dt)
                
                if data_str in val_str_list:
                    v_t = v_t.detach()
                    q_t = q_t.detach()
                    pv_t = pv_t.detach()
                
                if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                    console.print("[bold red]Encountered NaN values - stopping training[/bold red]")
                    break
                
                # Calculate RDF
                n_frames = q_t[::skip].shape[0]
                for idx in range(n_frames):
                    if idx == 0:
                        _, _, g_sim = rdf_obs_list[j](q_t[::skip][[idx]])
                    else:
                        g_sim += rdf_obs_list[j](q_t[::skip][[idx]])[2]
                
                g_sim = g_sim / n_frames
                
                # Calculate VACF
                vacf_sim = vacf_obs_list[j](v_t)
                
                # Calculate losses
                if data_str in data_str_list:
                    if vacf_target_list[j] is not None:
                        loss_vacf += (vacf_sim - vacf_target_list[j][:t_range]).pow(2).mean()
                    else:
                        loss_vacf += 0.0
                    
                    drdf = g_sim - rdf_target_list[j]
                    loss_rdf += (drdf).pow(2).mean()
                
                # Store results in logs
                obs_log[data_str]['rdf'].append(g_sim.detach().cpu().numpy())
                obs_log[data_str]['vacf'].append(vacf_sim.detach().cpu().numpy())
                
            # Calculate total loss
            if train_vacf == "True":
                loss = rdf_weight * loss_rdf + vacf_weight * loss_vacf
            else:
                loss = rdf_weight * loss_rdf
            
            # Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update scheduler
            scheduler.step(loss)
            
            # Save loss history
            loss_log.append([loss_vacf.item(), loss_rdf.item()])
            np.savetxt(f"{model_path}/loss.txt", np.array(loss_log), delimiter=',')
            
            # Check convergence
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr <= 1e-5:
                console.print("[bold green]Training converged![/bold green]")
                break
            
            # Update progress
            progress.update(train_task, advance=1)
    
    console.print("[bold green]Training completed![/bold green]")
    
    # Final plots
    for j, sim in enumerate(sim_list[:n_train]):
        data_str = (data_str_list + val_str_list)[j]
        console.print(f"[bold]Creating final plots for {data_str}...[/bold]")
        
        # Simulate one more time
        v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=dt)
        
        # Calculate final RDF
        n_frames = q_t[::skip].shape[0]
        for idx in range(n_frames):
            if idx == 0:
                _, _, g_sim = rdf_obs_list[j](q_t[::skip][[idx]])
            else:
                g_sim += rdf_obs_list[j](q_t[::skip][[idx]])[2]
        
        g_sim = g_sim / n_frames
        
        # Calculate final VACF
        vacf_sim = vacf_obs_list[j](v_t)
        
        # Plot final results
        if vacf_target_list[j] is not None:
            vacf_target = vacf_target_list[j][:t_range].detach().cpu().numpy()
        else:
            vacf_target = None
        
        rdf_target = rdf_target_list[j].detach().cpu().numpy()
        
        # Plot VACF
        plot_vacf(vacf_sim.detach().cpu().numpy(), vacf_target, 
                fn=f"{data_str}_final", 
                dt=dt,
                path=model_path)
        
        # Plot RDF
        rdf_start = rdf_obs_list[j].r_axis[0] if hasattr(rdf_obs_list[j], 'r_axis') else 0.75
        plot_rdf(g_sim.detach().cpu().numpy(), rdf_target, 
                fn=f"{data_str}_final",
                path=model_path, 
                start=rdf_start, 
                nbins=nbins,
                end=rdf_obs_list[j].r_axis[-1] if hasattr(rdf_obs_list[j], 'r_axis') else 2.5)
        
        # Plot final potential
        model_key = 'pair'  # Default model key
        if hasattr(sim.integrator.model, 'models'):
            if 'pair' in sim.integrator.model.models:
                model_key = 'pair'
            elif 'lj' in sim.integrator.model.models:
                model_key = 'lj'
                
        try:
            potential = plot_pair(
                path=model_path,
                fn="final",
                prior=sim.integrator.model.models[model_key].model, 
                device=device,
                target_pot=target_pot.to(device) if target_pot is not None else None,
                end=cutoff
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not plot final potential: {e}[/yellow]")
        

   # Display final optimized LJ parameters
    for j, sim in enumerate(sim_list[:n_train]):
        if hasattr(sim.integrator.model, 'models'):
            model_key = 'pair' if 'pair' in sim.integrator.model.models else 'lj'
            if model_key in sim.integrator.model.models:
                pair_pot = sim.integrator.model.models[model_key]
                if hasattr(pair_pot, 'model') and hasattr(pair_pot.model, 'sigma') and hasattr(pair_pot.model, 'epsilon'):
                    lj_model = pair_pot.model
                    console.print(f"\n[bold green]Final optimized Lennard-Jones parameters for {data_str_list[j]}:[/bold green]")
                    console.print(f"  σ (sigma) = {lj_model.sigma.item():.4f}")
                    console.print(f"  ε (epsilon) = {lj_model.epsilon.item():.4f}")
                    
                    # Optional: Save parameters to file
                    with open(f"{model_path}/optimized_parameters.txt", "a") as f:
                        f.write(f"System: {data_str_list[j]}\n")
                        f.write(f"Sigma: {lj_model.sigma.item():.6f}\n")
                        f.write(f"Epsilon: {lj_model.epsilon.item():.6f}\n")
                        f.write(f"Rep power: {lj_model.rep_pow}\n")
                        f.write(f"Attr power: {lj_model.attr_pow}\n")
                        f.write("-------------------\n")
        
    return loss_log, obs_log
