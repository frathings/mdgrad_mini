import os
import numpy as np
from ase import units
from md.sim import Simulations
from md.nhchain import NoseHooverChain
from data_src.data import load_config
from observables.rdf import rdf
from observables.observers import get_exp_rdf
from potential_src.pairMLP.potential_PairMLP import *
from potential_src.pairMLP.interface import PairPotentials, Stack

# Ensure the config.json file is present

# Load the configuration
config = load_config()

# Get data dictionaries
pair_data_dict = config.get("pair_data_dict", {})
exp_rdf_data_dict = config.get("exp_rdf_data_dict", {})



def get_observer(system, data_tag, nbins):

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


def get_sim(system, model, data_tag, topology_update_freq):

    T = exp_rdf_data_dict[data_tag]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T * units.kB,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=topology_update_freq).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim


def build_simulators(data_list, system_list, net, prior, cutoff, pair_flag, tpair_flag, topology_update_freq=1): 
    model_list = []
    for i, data_tag in enumerate(data_list):
        pair = PairPotentials(system_list[i], prior,
                        cutoff=cutoff,
                        ).to(system_list[i].device)

        if pair_flag:
            NN = PairPotentials(system_list[i], net,
                cutoff=cutoff,
                ).to(system_list[i].device)

        model = Stack({'nn': NN, 'pair': pair})
        model_list.append(model)

    sim_list = [get_sim(system_list[i], 
                        model_list[i], 
                        data_tag,
                        topology_update_freq) for i, data_tag in enumerate(data_list)]
    print('----------------------------------------------')
    print('----------------------------------------------')
    #print(NN)
    print(sim_list)
    print('----------------------------------------------')
    print('----------------------------------------------')
    return sim_list


