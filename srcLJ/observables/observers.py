import os
import numpy as np
from scipy import interpolate

from data_src.data import load_config
from observables.rdf import *


'''# Ensure the config.json file is present
config_file = '/Users/francescapagano/Documents/GitHub/mdml/srcLJ/config/config.json'
if not os.path.exists(config_file):
    print(f"Configuration file '{config_file}' not found.")
    
# Load the configuration
'''

config = load_config()



def get_exp_rdf(data, nbins, r_range, device, dim=3):
    # load RDF data 
    if data.shape[0] == 2:
        f = interpolate.interp1d(data[0], data[1])
    elif data.shape[1] == 2:
        f = interpolate.interp1d(data[:,0], data[:,1])

    start = r_range[0]
    end = r_range[1]
    xnew = np.linspace(start, end, nbins)
        
    # generate volume bins 
    V, vol_bins, _ = generate_vol_bins(start, end, nbins, dim=dim)
    vol_bins = vol_bins.to(device)

    g_obs = torch.Tensor(f(xnew)).to(device)
    g_obs_norm = ((g_obs.detach() * vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * vol_bins / V

    return xnew, g_obs

def get_observer(system, data_tag, nbins):
    # Validate if data_tag exists in the configuration
    exp_rdf_data_dict = config.get("exp_rdf_data_dict", {})
    if data_tag not in exp_rdf_data_dict:
        raise KeyError(f"Data tag '{data_tag}' not found in the configuration.")

    data_path = exp_rdf_data_dict[data_tag]['fn']
    data = np.loadtxt(data_path, delimiter=',')

    # define the equation of motion to propagate 
    start = exp_rdf_data_dict[data_tag]['start']
    end = exp_rdf_data_dict[data_tag]['end']

    xnew = np.linspace(start, end, nbins)

    obs = rdf(system, nbins, (start, end))
    # get experimental rdf 
    count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs.device)

    # initialize observable function 
    return xnew, g_obs, obs
