import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import sys 
import torch

from scipy import interpolate
from ase import units, Atoms

import json
import os
from scipy import interpolate
from ase import units
from ase.visualize import *
from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import json 
import matplotlib

# matplotlib.rcParams.update({'font.size': 25})
# matplotlib.rc('lines', linewidth=3, color='g')
# matplotlib.rcParams['axes.linewidth'] = 2.0
# matplotlib.rcParams['axes.linewidth'] = 2.0
# matplotlib.rcParams["xtick.major.size"] = 6
# matplotlib.rcParams["ytick.major.size"] = 6
# matplotlib.rcParams["ytick.major.width"] = 2
# matplotlib.rcParams["xtick.major.width"] = 2
# matplotlib.rcParams['text.usetex'] = False

#pair_data_dict = pair_data_dict.update(exp_rdf_data_dict)
pair_data_dict = {  'lj_0.3_1.2': {
            'rdf_fn': '../data/LJ_data/rdf_rho0.3_T1.2_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho0.3_T1.2_dt0.01.csv' ,
            'rho': 0.3,
            'T': 1.2, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },
  }

width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}

class LJFamily(torch.nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0, attr_pow=6,  rep_pow=12):
        super(LJFamily, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))
        self.attr_pow = attr_pow
        self.rep_pow = rep_pow 

    def LJ(self, r, sigma, epsilon):
        return 4 * epsilon * ((sigma/r)**self.rep_pow - (sigma/r)**self.attr_pow)

    def forward(self, x):
        return self.LJ(x, self.sigma, self.epsilon)


def plot_vacf(vacf_sim, vacf_target, fn, path, dt=0.01, save_data=False):

    t_range = np.linspace(0.0,  vacf_sim.shape[0], vacf_sim.shape[0]) * dt 

    plt.plot(t_range, vacf_sim, label='simulation', linewidth=4, alpha=0.6, )

    if vacf_target is not None:
        plt.plot(t_range, vacf_target, label='target', linewidth=2,linestyle='--', c='black' )

    plt.legend()
    plt.show()

    if save_data:
         np.savetxt(path + '/vacf_{}.txt'.format(fn), np.stack((t_range, vacf_sim)), delimiter=',' )
         np.savetxt(path + '/vacf_{}_target.txt'.format(fn), np.stack((t_range, vacf_target)), delimiter=',' )

    plt.savefig(path + '/vacf_{}.pdf'.format(fn), bbox_inches='tight')
    plt.close()

def plot_rdf( g_sim, rdf_target, fn, path, start, nbins, save_data=False, end=2.5):

    bins = np.linspace(start, end, nbins)

    plt.plot(bins, g_sim , label='simulation', linewidth=4, alpha=0.6)
    plt.plot(bins, rdf_target , label='target', linewidth=2,linestyle='--', c='black')
    
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")

    if save_data:
        np.savetxt(path + '/rdf_{}.txt'.format(fn), np.stack((bins, g_sim)), delimiter=',' )
        np.savetxt(path + '/rdf_{}_target.txt'.format(fn), np.stack((bins, rdf_target)), delimiter=',' )

    plt.show()
    plt.savefig(path + '/rdf_{}.pdf'.format(fn), bbox_inches='tight')
    plt.close()

def JS_rdf(g_obs, g):
    e0 = 1e-5
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js

def get_unit_len(rho, N_unitcell):
 
    L = (N_unitcell / rho) ** (1/3)
    
    return L 


def get_system(data_str, device, size):


    rho = pair_data_dict[data_str]['rho']
    T = pair_data_dict[data_str]['T']

    dim = pair_data_dict[data_str].get("dim", 3)

    if dim == 3:
        # initialize states with ASE 
        cell_module = pair_data_dict[data_str]['cell']
        N_unitcell = pair_data_dict[data_str]['N_unitcell']

        L = get_unit_len(rho, N_unitcell)

        print("lattice param:", L)

        atoms = cell_module(symbol=pair_data_dict[data_str]['element'],
                                  size=(size, size, size),
                                  latticeconstant= L,
                                  pbc=True)
        system = System(atoms, device=device)
        system.set_temperature(T)

    elif dim == 2:
        positions, cell = lattice_2d(rho, pair_data_dict[data_str]['size'])
        
        print("2D system")
        print("Number of atoms:{}".format(positions.shape[0]))
        print("Cell dim: \n {}".format(cell))

        atoms = Atoms(numbers=[1.] * positions.shape[0], positions=positions, cell=cell, pbc=True)
        system = System(atoms, device=device, dim=2)
        system.set_temperature(T)

    return system 

x

def get_target_obs(system, data_str, n_sim, rdf_range, nbins, t_range, dt, skip=25):
    
    print("simulating {}".format(data_str))
    device = system.device 
    
    target_pot = pair_data_dict[data_str]['target_pot']
    T = pair_data_dict[data_str]['T']

    pot = PairPotentials(system, target_pot, cutoff=2.5, nbr_list_device=device).to(device)

    diffeq = NoseHooverChain(pot, 
            system,
            Q=50.0, 
            T=T,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=1).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    rdf_obs = rdf(system, nbins=nbins, r_range=rdf_range)
    vacf_obs = vacf(system, t_range=t_range) 
    
    all_vacf_sim = []

    for i in range(n_sim):
        v_t, q_t, pv_t = sim.simulate(100, dt=dt, frequency=100)

        if i >= skip:
            vacf_sim = vacf_obs(v_t).detach().cpu().numpy()
            all_vacf_sim.append(vacf_sim)
            
    # loop over to ocmpute observables 
    trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device).detach()
    all_g_sim = []
    for i in range(len(trajs)):

        if i >= skip:
            _, _, g_sim = rdf_obs(trajs[[i]])
            all_g_sim.append(g_sim.detach().cpu().numpy())

    all_g_sim = np.array(all_g_sim).mean(0)
    all_vacf_sim = np.array(all_vacf_sim).mean(0)
    
    return all_g_sim, all_vacf_sim


def get_observer(system, data_str, nbins, t_range, rdf_start):

    # rdf_data_path = pair_data_dict[data_str]['rdf_fn']
    # rdf_data = np.loadtxt(rdf_data_path, delimiter=',')

    # if pair_data_dict[data_str].get("vacf_fn", None):
    #     vacf_data_path = pair_data_dict[data_str]['vacf_fn']
    #     vacf_target = np.loadtxt(vacf_data_path, delimiter=',')[:t_range]
    #     vacf_target = torch.Tensor(vacf_target).to(system.device)
    # else:
    #     vacf_target = None

    # get dt 
    dt = pair_data_dict[data_str].get('dt', 0.01)

    rdf_end = pair_data_dict[data_str].get("end", None)

    xnew = np.linspace(rdf_start , rdf_end, nbins)
        # initialize observable function 
    obs = rdf(system, nbins, (rdf_start , rdf_end) )
    vacf_obs = vacf(system, t_range=t_range) 

    # get experimental rdf 
    dim = pair_data_dict[data_str].get("dim", 3) 

    rdf_data_path = pair_data_dict[data_str].get("fn", None)
    # generate simulated data 
    if not rdf_data_path:
        rdf_data, vacf_target = get_target_obs(system, data_str, 200, (rdf_start, rdf_end), nbins=nbins, t_range=t_range, skip=50, dt=dt)
        vacf_target = torch.Tensor(vacf_target).to(system.device)
        rdf_data = np.vstack( (np.linspace(rdf_start, rdf_end, nbins), rdf_data))
    else:
        # experimental rdfs
        rdf_data = np.loadtxt(rdf_data_path, delimiter=',')
        vacf_target = None

    _, rdf_target = get_exp_rdf(rdf_data, nbins, (rdf_start, rdf_end), obs.device, dim=dim)

    # get model potential and simulate 

    return xnew, rdf_target, obs, vacf_target, vacf_obs

def get_sim(system, model, data_str, topology_update_freq=1):

    T = pair_data_dict[data_str]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=topology_update_freq).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim

def plot_pair(fn, path, model, prior, device, end=2.5, target_pot=None): 

    if target_pot is None:
        target_pot = LennardJones(1.0, 1.0)
    else:
        target_pot = target_pot.to("cpu")

    x = torch.linspace(0.1, end, 250)[:, None].to(device)
    
    u_fit = (model(x) + prior(x)).detach().cpu().numpy()
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
    plt.show()
    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

    return u_fit

def fit_lj(assignments, suggestion_id, device, sys_params, project_name):

    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    size = sys_params['size']
    cutoff = sys_params['cutoff']
    t_range = sys_params['t_range']

    nbins = assignments['nbins']
    tau = assignments['opt_freq']

    rdf_start = assignments.get("start", 0.75)
    skip = 1

    nbr_list_device = sys_params.get("nbr_list_device", device)
    topology_update_freq = sys_params.get("topology_update_freq", 1)

    data_str_list = sys_params['data']

    # Get the grounth truth pair potentials
    target_pot = pair_data_dict[data_str_list[0]].get("target_pot", None)

    # merge paramset 
    if sys_params['val']:
        val_str_list = sys_params['val']
    else:
        val_str_list = []

    print(json.dumps(assignments, indent=1))

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    # merge paramset 
    paramset = {**sys_params, **assignments}
    # dump paramset 
    with open(model_path + '/paramset.json', 'w') as fp:
        json.dump(paramset, fp, indent=4)

    print("Training for {} epochs".format(n_epochs))

    system_list = []
    for data_str in data_str_list + val_str_list:
        system = get_system(data_str, device, size) 
        system_list.append(system)

    # Define prior potential
    mlp_parmas = {'n_gauss': int(cutoff//assignments['gaussian_width']), 
              'r_start': 0.0,
              'r_end': cutoff, 
              'n_width': assignments['n_width'],
              'n_layers': assignments['n_layers'],
              'nonlinear': assignments['nonlinear']}

    lj_params = {'epsilon': assignments['epsilon'], 
         'sigma': assignments['sigma'],
        "power": assignments['power']}

    NN = pairMLP(**mlp_parmas)
    pair = LJFamily(epsilon=2.0, sigma=assignments['sigma'], rep_pow=6, attr_pow=3)  # ExcludedVolume(**lj_params)

    model_list = []
    for i, data_str in enumerate(data_str_list + val_str_list):

        pairNN = PairPotentials(system_list[i], NN,
                    cutoff=cutoff,
                    nbr_list_device=nbr_list_device
                    ).to(device)
        prior = PairPotentials(system_list[i], pair,
                        cutoff=2.5,
                    nbr_list_device=nbr_list_device
                        ).to(device)

        model = Stack({'pairnn': pairNN, 'pair': prior})
        model_list.append(model)


    sim_list = [get_sim(system_list[i], 
                        model_list[i], 
                        data_str,
                        topology_update_freq=topology_update_freq) for i, data_str in enumerate(data_str_list + val_str_list)]

    

    nbins = assignments['nbins']

    rdf_obs_list = []
    vacf_obs_list = []

    rdf_target_list = []
    vacf_target_list = []
    rdf_bins_list = []

    for i, data_str in enumerate(data_str_list + val_str_list):
        rdf_start = pair_data_dict[data_str].get("start", 0.75)
        x, rdf_target, rdf_obs, vacf_target, vacf_obs = get_observer(system_list[i],
                                                                     data_str, 
                                                                     nbins, 
                                                                     t_range=t_range,
                                                                     rdf_start=rdf_start)
        rdf_bins_list.append(x)

        rdf_obs_list.append(rdf_obs)
        rdf_target_list.append(rdf_target)
        vacf_obs_list.append(vacf_obs)
        vacf_target_list.append(vacf_target)

    
    optimizer = torch.optim.Adam(list(NN.parameters()), lr=assignments['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=1e-6, 
                                                  verbose=True, factor = 0.5, patience= 20,
                                                  threshold=5e-5)

    # Set up simulations 
    loss_log = []

    # 
    obs_log = dict()

    for i, data_str in enumerate(data_str_list + val_str_list):
        obs_log[data_str] = {}
        obs_log[data_str]['rdf'] = []
        obs_log[data_str]['vacf'] = []


    for i in range(sys_params['n_epochs']):

        loss_rdf = torch.Tensor([0.0]).to(device)
        loss_vacf = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        n_train = len(data_str_list)
        for j, sim in enumerate(sim_list[:n_train]): # only simulate config that needs training 

            data_str = (data_str_list + val_str_list)[j]

            # get dt 
            dt = pair_data_dict[data_str].get("dt", 0.01)

            # Simulate 
            v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=dt)

            if data_str in val_str_list:
                v_t = v_t.detach()
                q_t = q_t.detach()
                pv_t = pv_t.detach()

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                print("encounter NaN")
                return 5 - (i / n_epochs) * 5

            #_, _, g_sim = rdf_obs_list[j](q_t[::skip])

            # save memory by computing it in serial
            skip = 5
            n_frames = q_t[::skip].shape[0] 
            for idx in range(n_frames):
                if idx == 0:
                    _, _, g_sim = rdf_obs_list[j](q_t[::skip][[idx]])
                else:
                    g_sim += rdf_obs_list[j](q_t[::skip][[idx]])[2]

            g_sim = g_sim / n_frames

            # compute vacf 
            vacf_sim = vacf_obs_list[j](v_t)

            if data_str in data_str_list:
                if vacf_target_list[j] is not None:
                    loss_vacf += (vacf_sim - vacf_target_list[j][:t_range]).pow(2).mean()
                else:
                    loss_vacf += 0.0

                drdf = g_sim - rdf_target_list[j]
                loss_rdf += (drdf).pow(2).mean()#+ JS_rdf(g_sim, rdf_target_list[j])

            obs_log[data_str]['rdf'].append(g_sim.detach().cpu().numpy())
            obs_log[data_str]['vacf'].append(vacf_sim.detach().cpu().numpy())

            if i % 5 ==0 :
                if vacf_target_list[j] is not None:
                    vacf_target = vacf_target_list[j][:t_range].detach().cpu().numpy()
                else:
                    vacf_target = None
                rdf_target = rdf_target_list[j].detach().cpu().numpy()

                plot_vacf(vacf_sim.detach().cpu().numpy(), vacf_target, 
                    fn=data_str + "_{}".format(str(i).zfill(3)), 
                    dt=dt,
                    path=model_path)

                plot_rdf(g_sim.detach().cpu().numpy(), rdf_target, 
                    fn=data_str + "_{}".format(str(i).zfill(3)),
                     path=model_path, 
                     start=rdf_start, 
                     nbins=nbins,
                     end=rdf_obs_list[j].r_axis[-1])

            if i % 5 ==0 :
                potential = plot_pair( path=model_path,
                             fn=str(i).zfill(3),
                              model=sim.integrator.model.models['pairnn'].model, 
                              prior=sim.integrator.model.models['pair'].model, 
                              device=device,
                              target_pot=target_pot.to(device),
                              end=cutoff)

        if assignments['train_vacf'] == "True":
            loss = assignments['rdf_weight'] * loss_rdf + assignments['vacf_weight'] * loss_vacf
        else:
            loss = assignments['rdf_weight'] * loss_rdf

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        print(loss_vacf.item(), loss_rdf.item())
        
        scheduler.step(loss)
        
        loss_log.append([loss_vacf.item(), loss_rdf.item() ])

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 1e-5:
            print("training converged")
            break

        np.savetxt(model_path + '/loss.txt', np.array(loss_log), delimiter=',')
    
    # # save potentials         
    # if np.array(loss_log)[-10:, 1].mean() <=  0.005: 
    #     np.savetxt(model_path + '/potential.txt',  potential, delimiter=',')

    rdf_dev = []

    for j, sim in enumerate(sim_list):

        #simulate with no optimization
        data_str = (data_str_list + val_str_list)[j]

        dt = pair_data_dict[data_str].get("dt", 0.01)

        all_vacf_sim = []

        for i in range(sys_params['n_sim']):
            v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=dt)

            # compute VACF 
            vacf_sim = vacf_obs_list[j](v_t).detach().cpu().numpy()
            all_vacf_sim.append(vacf_sim)

        all_vacf_sim = np.array(all_vacf_sim).mean(0)
        
        trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device).detach()

        # get targets
        if vacf_target_list[j] is not None:
            vacf_target = vacf_target_list[j][:t_range].detach().cpu().numpy()
        else:
            vacf_target = None
        rdf_target = rdf_target_list[j].detach().cpu().numpy()
        

        # loop over to ocmpute observables 
        all_g_sim = []
        for i in range(len(trajs)):
            _, _, g_sim = rdf_obs_list[j](trajs[[i]])
            all_g_sim.append(g_sim.detach().cpu().numpy())

        all_g_sim = np.array(all_g_sim).mean(0)
        
        # compute target deviation 
        if data_str in data_str_list:
            drdf = np.abs(all_g_sim - rdf_target_list[j].cpu().numpy()).mean()
            rdf_dev.append(drdf) 

        # plot observables 
        plot_vacf(all_vacf_sim, vacf_target, 
            fn=data_str, 
            path=model_path,
            dt=dt,
            save_data=True)

        plot_rdf(all_g_sim, rdf_target, 
            fn=data_str,
             path=model_path, 
             start=rdf_start, 
             nbins=nbins,
             save_data=True,
             end=rdf_obs_list[j].r_axis[-1])


        # rdf_dev = np.abs(all_g_sim - rdf_target).mean()

    np.savetxt(model_path + '/potential.txt',  potential, delimiter=',')
    np.savetxt(model_path + '/rdf_dev.txt', np.array(rdf_dev), delimiter=',')


    # save loss curve 
    plt.plot(np.array( loss_log)[:, 0], label='vacf', alpha=0.7)
    plt.plot(np.array( loss_log)[:, 1], label='rdf', alpha=0.7)
    plt.yscale("log")
    plt.legend()
    
    plt.savefig(model_path + '/loss.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    

    return np.array(loss_log)[-10:, 1].mean() 


import argparse
from datetime import datetime
from datetime import date
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=None)
parser.add_argument("-nruns", type=int, default=1)

parser.add_argument("-paramset", type=str, default='None')

# options to overide default param set
parser.add_argument("-sigma", type=float)
parser.add_argument("-lr", type=float)
parser.add_argument("-cutoff", type=float)
parser.add_argument("-vacf_weight", type=float)
parser.add_argument("-dt", type=float)
parser.add_argument("-update_freq", type=int)
parser.add_argument("-opt_freq", type=int, default=120)

parser.add_argument("-name", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--trainvacf", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    n_obs = 2
    n_epochs = 4
    n_sim = 2
else:
    n_obs = 1000
    n_epochs = 1000    # 1000
    n_sim = 200   # 50

logdir = params['logdir']


if params['paramset'] != 'None':
    import json
    assignments = json.load(open(params['paramset']))

else:
    # use default assignments
    assignments = {
        "epsilon": 0.4,
        "gaussian_width": 0.10,
        "n_layers": 3,
        "n_width": 128,
        "nbins": 100,
        "nonlinear": "ELU",
        "opt_freq": params['opt_freq'],   # 60
        "power": 10,
        "rdf_weight": 0.95,
        "train_vacf": "True",
        "lr": 0.002,
        "sigma": 0.9,
        "vacf_weight": 0.41,
        "cutoff": 2.5
        }

sys_params = {
    'dt': params['dt'],
    'size': 4,
    'n_epochs': n_epochs,
    'n_sim': n_sim,
    'data': params['data'],
    'val': params['val'],
    't_range': 50,   # 50
    'skip': 5,
    'cutoff': assignments['cutoff'],
    'nbr_list_device': 'cpu',
    'topology_update_freq': params['update_freq']
}

if assignments['train_vacf'] == 'False':
    assignments['vacf_weight'] = 0.0

# param overwrite
if params['trainvacf']:
    assignments['train_vacf'] = 'True'

if params['sigma'] is not None:
    print("chaging default sigma from {} to {}".format(assignments['sigma'], params['sigma']))
    assignments['sigma'] = params['sigma']

if params['lr'] is not None:
    print("chaging default lr from {} to {}".format(assignments['lr'], params['lr']))
    assignments['lr'] = params['lr']

if params['cutoff'] is not None:
    print("chaging default cutoff from {} to {}".format(assignments['cutoff'], params['cutoff']))
    sys_params['cutoff'] = params['cutoff']

if params['vacf_weight'] is not None:
    print("chaging default vacf_weight from {} to {}".format(assignments['vacf_weight'], params['vacf_weight']))
    assignments['vacf_weight'] = params['vacf_weight']

if params['device'] is None:
    params['device'] = 'cpu'

print(assignments)

for i in range(params['nruns']):

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

    value = fit_lj(assignments=assignments,
                   suggestion_id=params['name'] + dt_string,
                   device=params['device'],
                   sys_params=sys_params,
                   project_name=logdir)

# show test error
print(value)