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

def plot_rdfs(bins, target_g, simulated_g, fname, path, pname=None, save=True):
    plt.title("epoch {}".format(pname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()
    if save:
        data = np.vstack((bins, simulated_g.detach().cpu().numpy()))
        np.savetxt(path + '/{}.csv'.format(fname), data, delimiter=',')

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = ase.Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

def plot_pair(fn, path, model, prior, device, start=0.5, end=2.5): 

    x = torch.linspace(start, end, 1000)[:, None].to(device)
    
    u_fit = (model(x) + prior(x)).detach().cpu().numpy()
    u_fit = u_fit - u_fit[-1] 

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)

    #plt.ylabel("g(r)")
    plt.legend()      
    plt.show()
    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

    return u_fit

assignments = {
        "anneal_freq": 7,
        "anneal_rate": 5.2,
        "cutoff": 4.9,
        "epsilon": 1,
        "gaussian_width": 0.145,
        "n_width": 128,
        "n_layers": 3,
        "lr": 0.000025,
        "mse_weight": 0.4,
        "n_atom_basis": "high",
        "n_convolutions": 3,
        "n_filters": "mid",
        "nbins": 90,
        "opt_freq": 26,
        "sigma": 1.9,
        "start_T": 200,
        "nonlinear": "ReLU",
        "power": 12,
    }

sys_params = {
        'dt': 1.0,
        'n_epochs': 500,
        'n_sim': 10,
        'data': "H20_0.997_298K", #'Si_2.293_100K',
        'val': None,
        'size': 4,
        'anneal_flag': True,
        'pair_flag': True,
    }
config = load_config()

# Get data dictionaries
pair_data_dict = config.get("pair_data_dict", {})
exp_rdf_data_dict = config.get("exp_rdf_data_dict", {})

def init_sys():
    config = load_config()
    data_tag =   "H20_0.997_298K"   #"Si_2.293_100K"  # Change this to a data tag from config.json
    size = sys_params['size']  # System size for visualization
    # Check if the data tag exists in the configuration
    if data_tag not in config["exp_rdf_data_dict"]:
        print(f"Data tag '{data_tag}' not found in config.")
        return
    # Run visualization
    print(f"Visualizing system for data tag '{data_tag}'...")
    visualize_system_with_ase_3d(data_tag, size)
    print("Visualization completed.")

def make_folder(project_name, suggestion_id):
    """
    Creates a folder structure for the given project and suggestion ID.
    Only creates the folder if it does not already exist.
    """
    model_path = '{}/{}'.format(project_name, suggestion_id)
    if not os.path.exists(model_path):  # Check if the folder exists
        os.makedirs(model_path)  # Create the folder if it doesn't exist
    return model_path

def training_setup_sys(model_path, device, make_folder_flag = False):
    data_tag = "H20_0.997_298K"#"Si_2.293_100K" 
    if make_folder_flag:
        make_folder('test', '1')
    print("Training for {} epochs".format(sys_params['n_epochs']))
    train_list = ['H20_0.997_298K'] # = 'H20_0.997_298K'     'H20_0.997_298K': { 'fn': "../data/water_exp/water_exp_pccp.csv",
                                                                                #   'rho': 0.997,
                                                                                #   'T': 298.0, 
                                                                                #   'start': 1.8, 
                                                                                #   'end': 7.5,
                                                                                #   'element': "H" ,
                                                                                #   'mass': 18.01528,
                                                                                #   "N_unitcell": 8,
                                                                                #   "cell": Diamond, #FaceCenteredCubic
                                                                                #   "pressure": 1.0 # MPa
                                                                                #   },
    print('Train_list: ', train_list)                                                        
    if sys_params['val']:
        all_sys = train_list + sys_params['val']
    else:
        all_sys = train_list

    system_list = []
    for data_tag in all_sys:
        if data_tag not in exp_rdf_data_dict:
            raise KeyError(f"Data tag '{data_tag}' not found in exp_rdf_data_dict.")
        system = get_system(data_tag, device, sys_params['size'])

    for data_tag in all_sys:
        #print('data tag: ',data_tag) #H20_0.997_298K
        system = get_system(data_tag, device, sys_params['size']) 
        if sys_params['anneal_flag'] == 'True': # set temp 
            system.set_temperature(assignments['start_T'] * units.kB)
        system_list.append(system)

    if sys_params["pair_flag"]:
        net, prior = get_pair_potential(assignments, sys_params)


        def pair_pretrain(all_sys, net, prior):
            # u_target: pot = - units.kB * T * torch.log(g_obs) + averaged + interpolated
            # ------ code to pretrain -----
            net = net.to(device)
            prior = prior.to(device)

            all_pot = []
            for i, data_tag in enumerate(all_sys):
                x, g_obs, obs = get_observer(system_list[i], data_tag, assignments['nbins'])
                print('x, g_obs, g: ', x, g_obs, obs )
                print('-----------------------------')
                T = exp_rdf_data_dict[data_tag]['T']
                pot = - units.kB * T * torch.log(g_obs)
                all_pot.append(pot)

            bi = torch.stack(all_pot).mean(0)
            bi = torch.nan_to_num(bi,  posinf=100.0)

            f = interpolate.interp1d(x, bi.detach().cpu().numpy())
            rrange = np.linspace(2.5, 7.5, 1000)
            u_target = f(rrange)

            u_target = torch.Tensor(u_target).to(device)
            rrange = torch.Tensor(rrange).to(device)

            optimizer = torch.optim.Adam(list(net.parameters()), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                      'min', 
                                                      min_lr=0.9e-7, 
                                                      verbose=True, factor = 0.5, patience=25,
                                                      threshold=1e-5)
            
            for i in range(4000):
                u_fit = net(rrange.unsqueeze(-1)) + prior(rrange.unsqueeze(-1))
                loss = (u_fit.squeeze() - u_target).pow(2).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step(loss.item())

                if i % 50 == 0:
                    print(i, loss.item())

            np.savetxt(model_path + f'/bi.txt', u_target.detach().cpu().numpy())
            np.savetxt(model_path + f'/fit.txt', u_fit.detach().cpu().numpy())
            print('esco')
    
    pair_pretrain(all_sys, net, prior) 

    sim_list = build_simulators(all_sys, system_list, net, prior, 
                                cutoff=assignments['cutoff'], pair_flag=sys_params["pair_flag"],
                                tpair_flag=False,
                                topology_update_freq=1)

    g_target_list = []
    obs_list = []
    bins_list = []

    for i, data_tag in enumerate(all_sys):
        x, g_obs, obs = get_observer(system_list[i], data_tag, assignments['nbins'])
        bins_list.append(x)
        g_target_list.append(g_obs)
        obs_list.append(obs)
    #plt.plot(x,g_obs)
    #plt.plot(x,obs)
    #plt.title('g_obs')
    #plt.show()
    # define optimizer 
    optimizer = torch.optim.Adam(list(net.parameters()), lr=assignments['lr'])

    loss_log = []

    solver_method = 'NH_verlet'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=0.9e-7, 
                                                  verbose=True, factor = 0.5, patience=25,
                                                  threshold=1e-5)

    for i in range(0, sys_params['n_epochs'] ):
        print('here')    

        loss_js = torch.Tensor([0.0]).to(device)
        loss_mse = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        for j, sim in enumerate(sim_list[:len(train_list)]):

            data_tag = all_sys[j]

            if sys_params['anneal_flag'] == 'True' and i % assignments['anneal_freq'] == 0:

                T_equil = exp_rdf_data_dict[data_tag]['T']
                T_start = assignments['start_T']
                new_T  = get_temp(T_start, T_equil, sys_params['n_epochs'] , i, assignments['anneal_rate'])
                sim.integrator.update_T(new_T * units.kB)

                print("update T: {:.2f}".format(new_T))

            v_t, q_t, pv_t = sim.simulate(steps=assignments['opt_freq'] , frequency=int(assignments['opt_freq'] ))

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5 - (i / sys_params['n_epochs'] ) * 5

            _, bins, g = obs_list[j](q_t[::20])
        
        #---------------------------------------------------------------------
            # only optimize on data that needs training 
            if data_tag in train_list:

                def compute_D(dev, rho, rrange):
                    return (4 * np.pi * rho * (rrange ** 2) * dev ** 2 * (rrange[2]- rrange[1])).sum()

                loss_js += JS_rdf(g_target_list[j], g)
                #loss_mse += assignments['mse_weight'] * (g - g_target_list[j]).pow(2).mean() 

                rrange = torch.linspace(bins[0], bins[-1], g.shape[0])
                rho = system_list[j].get_number_of_atoms() / system_list[j].get_volume()

                loss_mse += compute_D(g - g_target_list[j], rho, rrange.to(device))

            if i % 10 == 0:
                plot_rdfs(bins_list[j], g_target_list[j], g, "{}_{}".format(data_tag, i),
                             model_path, pname=i)

                if sys_params['pair_flag']:
                    potential = plot_pair( path=model_path,
                                 fn=str(i),
                                  model=net, 
                                  prior=prior, 
                                  device=device,
                                  start=2, end=8)

                    np.savetxt(model_path + '/potential.txt', potential)

        #--------------------------------------------------------------------------------

        loss = loss_mse 
        loss.backward()
        
        print("epoch {} | loss: {:.5f}".format(i, loss.item()) ) 
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        loss_log.append(loss_js.item() )

        if optimizer.param_groups[0]["lr"] <= 1.0e-5:
            print("training converged")
            break

    plt.plot(loss_log)
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    total_loss = 0.
    rdf_devs = []
    for j, sim in enumerate(sim_list):    
        data_tag = all_sys[j]

        train_traj = sim.log['positions']

        if all_sys[j] in train_list:
            save_traj(system_list[j], train_traj, model_path + '/{}_train.xyz'.format(data_tag), skip=10)
        else:
            save_traj(system_list[j], train_traj, model_path + '/{}_val.xyz'.format(data_tag), skip=10)

        # Inference 

        for i in range(sys_params['n_sim'] ):
            _, q_t, _ = sim.simulate(steps=100, frequency=25)
            
        trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device)

        test_nbins = 800
        x, g_obs, obs = get_observer(system_list[j], data_tag, test_nbins)

        all_g_sim = []
        for i in range(len(trajs)):
            _, _, g_sim = obs(trajs[[i]])
            all_g_sim.append(g_sim.detach().cpu().numpy())

        all_g_sim = np.array(all_g_sim).mean(0)

        # compute equilibrated rdf 
        loss_js = JS_rdf(g_obs, torch.Tensor(all_g_sim).to(device))

        loss_mse = (g_obs - torch.Tensor(all_g_sim).to(device)).pow(2).mean()

        if data_tag in train_list:
            rdf_devs.append( (g_obs - torch.Tensor(all_g_sim).to(device)).abs().mean().item())

        save_traj(system_list[j], np.stack( sim.log['positions']),  
            model_path + '/{}_sim.xyz'.format(data_tag), skip=1)

        plot_rdfs(x, g_obs, torch.Tensor(all_g_sim), "{}_final".format(data_tag), model_path, pname='final')

        total_loss += loss_mse.item()

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))
    np.savetxt(model_path + '/rdf_mse.txt', np.array(rdf_devs))

    return total_loss

if __name__ == "__main__":
    try:
        init_sys()
        make_folder('test', '1')
        training_setup_sys('test/1', 'cpu')
    except Exception as e:
        print(f"Error occurred: {e}")
