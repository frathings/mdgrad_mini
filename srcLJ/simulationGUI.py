import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *
from tqdm import tqdm

import matplotlib.pyplot as plt



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


class SimulationGUI:
    def __init__(self, root, main_app):
        """Initialize the Simulation GUI as a child window."""
        self.root = root
        self.root.title("Simulation GUI")

        self.main_app = main_app  # Reference to MainGUI
        self.selected_model = None
        self.selected_integrator = "AndersenODE"  # Default integrator

        # Model Selection Dropdown
        tk.Label(root, text="Select Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, state="readonly")
        self.model_dropdown.grid(row=0, column=1)

        # Integrator Selection Dropdown
        tk.Label(root, text="Select Integrator:").grid(row=1, column=0, sticky="w")
        self.integrator_var = tk.StringVar(value="AndersenODE")
        self.integrator_dropdown = ttk.Combobox(root, textvariable=self.integrator_var, state="readonly",
                                                values=["AndersenODE", "NoseHooverChain"])
        self.integrator_dropdown.grid(row=1, column=1)

        # Run Simulation Button
        self.run_sim_btn = tk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_sim_btn.grid(row=2, column=0, columnspan=2)

        # Output Box
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.grid(row=3, column=0, columnspan=2)

        # Load Available Models
        self.load_models()
        
        self.train_vacf = False
        
        # Observable Storage
        self.rdf_obs_list = []  # Stores functions to compute RDF during training.
        self.vacf_obs_list = []  # Stores functions to compute VACF during training.
        self.rdf_target_list = []  # Stores ground truth RDF data for comparison.
        self.vacf_target_list = []  # Stores ground truth VACF data for comparison.
        self.rdf_bins_list = []  # Stores distance bins for RDF computation.
        self.nbins = 100
        self.t_range = 50

        self.optimizer = torch.optim.Adam(list(self.main_app.NN.parameters()), lr=0.002)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        'min', 
                                                        min_lr=1e-6, 
                                                        verbose=True, factor = 0.5, patience= 20,
                                                        threshold=5e-5)

        # Set up simulations 
        self.loss_log = []
        self.sim = None
        self.model_path = 'test_GUI'
        self.obs_log = dict()
        self.rdf_start = 0.75

        self.rdf_weight = 0.95
        self.vacf_weight = 0.0

    def load_models(self):
        """Load models from MainGUI into the dropdown."""
        if not self.main_app.model_list:
            messagebox.showwarning("Warning", "No models available! Load models first.")
            return

        model_names = [f"Model {i+1}" for i in range(len(self.main_app.model_list))]
        self.model_dropdown["values"] = model_names
        if model_names:
            self.model_var.set(model_names[0])  # Default selection

    def get_sim(self, system, model, data_str, integrator="AndersenODE", topology_update_freq=1):
        """Initialize a simulation with the selected integrator."""
        T = self.main_app.pair_data_dict[data_str]['T']

        if integrator == "AndersenODE":
            print('AAAAAAAAAAAAAAS')
            dt = 1e-3  # Set an appropriate time step.
            nu = 100     # Set an appropriate collision frequency.

            integrator_diffeq = AndersenODE(
                potentials=model,
                system=system,
                T=T,
                dt=dt,
                nu=nu
            ).to(system.device)
            sim = Simulations(system, integrator_diffeq, method = 'verlet')

        elif integrator == "NoseHooverChain":
            print('NNNNNNNNNNNN')
            integrator_diffeq = NoseHooverChain(
                model, 
                system,
                Q=50.0,
                T=T,
                num_chains=5,
                adjoint=True,
                topology_update_freq=topology_update_freq
            ).to(system.device)
            sim = Simulations(system, integrator_diffeq, method = 'NH_verlet')

        else:
            raise ValueError(f"Invalid integrator: {integrator}")

        # Create the simulation
        
        return sim

    def run_simulation(self):
        """Run simulation with the selected model and integrator."""
        selected_index = self.model_dropdown.current()
        if selected_index == -1:
            messagebox.showerror("Error", "Please select a model first.")
            return

        self.selected_model = self.main_app.model_list[selected_index]
        selected_integrator = self.integrator_var.get()
        system = self.main_app.system_list[selected_index]
        data_str = self.main_app.data_str_list[selected_index]

        messagebox.showinfo("Running", f"Running simulation with {selected_integrator}...")

        try:
            self.sim = self.get_sim(system, self.selected_model, data_str, integrator=selected_integrator, topology_update_freq=1)

            print(' ========= Initializing Observables ========= ')
            rdf_start = self.main_app.pair_data_dict[data_str].get("start", 0.75)

            x, rdf_target, rdf_obs, vacf_target, vacf_obs = self.get_observer(system, data_str, self.nbins, t_range=self.t_range, rdf_start=rdf_start)

            self.rdf_bins_list.append(x)
            self.rdf_obs_list.append(rdf_obs)
            self.rdf_target_list.append(rdf_target)
            self.vacf_obs_list.append(vacf_obs)
            self.vacf_target_list.append(vacf_target)

            # Convert tensor to NumPy array
            rdf_target = self.rdf_target_list[0].cpu().numpy()  # Assuming first entry is needed
            plt.figure(figsize=(8, 5))
            plt.plot(self.rdf_bins_list[0], rdf_target, label="RDF Target", color='b', linewidth=2)

            plt.xlabel("Distance (Å)")
            plt.ylabel("g(r)")
            plt.title("Radial Distribution Function (RDF)")
            plt.legend()
            plt.grid()
            plt.show()

            #for i, data_str in enumerate(data_str_list + val_str_list):
            self.obs_log[data_str] = {}
            self.obs_log[data_str]['rdf'] = []
            self.obs_log[data_str]['vacf'] = []

            self.train()

        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
        
    def get_observer(self, system, data_str, nbins, t_range, rdf_start):
        """Initialize observables and retrieve ground truth RDF data."""
        dt = self.main_app.pair_data_dict[data_str].get('dt', 0.01)
        rdf_end = self.main_app.pair_data_dict[data_str].get("end", None)

        xnew = np.linspace(rdf_start, rdf_end, nbins)

        # Observable function
        obs = rdf(system, nbins, (rdf_start, rdf_end))
        vacf_obs = vacf(system, t_range=t_range)

        # Retrieve experimental RDF data
        rdf_data_path = self.main_app.pair_data_dict[data_str].get("fn", None)
        print('starting...')
        if not rdf_data_path:
            rdf_data, vacf_target = self.get_target_obs(system, data_str, 100, (rdf_start, rdf_end), nbins, t_range, skip=50, dt=dt)
            vacf_target = torch.Tensor(vacf_target).to(system.device)
            rdf_data = np.vstack((np.linspace(rdf_start, rdf_end, nbins), rdf_data))
        else:
            rdf_data = np.loadtxt(rdf_data_path, delimiter=',')
            vacf_target = None

        _, rdf_target = get_exp_rdf(rdf_data, nbins, (rdf_start, rdf_end), obs.device)

        return xnew, rdf_target, obs, vacf_target, vacf_obs

    def get_target_obs(self, system, data_str, n_sim, rdf_range, nbins, t_range, dt, skip=25, integrator="AndersenODE"):
        """Simulate observables using the target potential."""
        print('starting....')
        device = system.device
        target_pot = LennardJones()
        T = self.main_app.pair_data_dict[data_str]['T']
        print('targetpot: ', target_pot)
        pot = PairPotentials(system, target_pot, cutoff=2.5, nbr_list_device=device).to(device)

        if integrator == "NoseHooverChain":
            #print('nhh')
            integrator_diffeq = NoseHooverChain(pot, system, Q=50.0, T=T, num_chains=5, adjoint=True, topology_update_freq=1).to(system.device)
            self.sim = Simulations(system, integrator_diffeq, method='NH_verlet')

        elif integrator == "AndersenODE":
            dt = 1e-3  # Set an appropriate time step.
            nu = 100     # Set an appropriate collision frequency.
            integrator_diffeq = AndersenODE(
                potentials=pot,
                system=system,
                T=T,
                dt=dt,
                nu=nu
            ).to(system.device)
            self.sim = Simulations(system, integrator_diffeq, method='verlet')
        
        # define objects for the observables
        rdf_obs = rdf(system, nbins=nbins, r_range=rdf_range)
        vacf_obs = vacf(system, t_range=t_range) 
        
        all_vacf_sim = []
        self.t_range = 50

        self.root.update_idletasks()  # Ensure UI updates

        # Run MD Simulations
        for i in tqdm(range(n_sim), desc="Running Simulation", unit="step"):
            if integrator == "NoseHooverChain":
                #print('here')
                v_t, q_t, pv_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.
            elif integrator == "AndersenODE":
                #print('here now')
                v_t, q_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.

            if i >= skip:
                vacf_sim = vacf_obs(v_t).detach().cpu().numpy()
                all_vacf_sim.append(vacf_sim)

        # loop over to compute observables 
        trajs = torch.Tensor( np.stack( self.sim.log['positions'])).to(system.device).detach()
        all_g_sim = []
        for i in range(len(trajs)):

            if i >= skip:
                _, _, g_sim = rdf_obs(trajs[[i]])
                all_g_sim.append(g_sim.detach().cpu().numpy())

        all_g_sim = np.array(all_g_sim).mean(0)
        all_vacf_sim = np.array(all_vacf_sim).mean(0)
        
        return all_g_sim, all_vacf_sim
    
    def train(self):
        n_epochs = 5
        device = 'cpu'
        cutoff = 2.5
        selected_index = self.model_dropdown.current()
        if selected_index == -1:
            messagebox.showerror("Error", "Please select a model first.")
            return
        

        self.selected_model = self.main_app.model_list[selected_index]
        print('models: ', self.selected_model)
        selected_integrator = self.integrator_var.get()
        system = self.main_app.system_list[selected_index]
        data_str = self.main_app.data_str_list[selected_index]
        self.sim = self.get_sim(system, self.selected_model, data_str, integrator=selected_integrator, topology_update_freq=1)
        target_pot = LennardJones()

        tau = 60
        print('in train')
        for i in range(n_epochs):
            print('epoch n. ',i)
            loss_rdf = torch.Tensor([0.0]).to(device)
            loss_vacf = torch.Tensor([0.0]).to(device)

            n_train =  1
            sim_list = [self.sim]
            
            for j, sim in enumerate(sim_list[:n_train]): 
                print('================================================')
                #data_str = (data_str_list + val_str_list)[j]
                # get dt 
                dt = 0.005
                # Simulate 
                if selected_integrator == "AndersenODE":
                    v_t, q_t = sim.simulate(steps=tau, frequency=tau, dt=dt)
                    
                elif selected_integrator == "NoseHooverChain":
                    v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=dt)
                    
                print('passato')

                if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                    print("encounter NaN")
                    print( 5 - (i / n_epochs) * 5 )
                    break
                #_, _, g_sim = rdf_obs_list[j](q_t[::skip])
                # save memory by computing it in serial
                skip = 5
                n_frames = q_t[::skip].shape[0] 
                for idx in range(n_frames):
                    if idx == 0:
                        _, _, g_sim = self.rdf_obs_list[j](q_t[::skip][[idx]])
                    else:
                        g_sim += self.rdf_obs_list[j](q_t[::skip][[idx]])[2]

                g_sim = g_sim / n_frames

                # compute vacf 
                vacf_sim = self.vacf_obs_list[j](v_t)

                if data_str in self.main_app.data_str_list:
                    if self.vacf_target_list[j] is not None:
                        loss_vacf += (vacf_sim - self.vacf_target_list[j][:self.t_range]).pow(2).mean()
                    else:
                        loss_vacf += 0.0

                    drdf = g_sim - self.rdf_target_list[j]
                    loss_rdf += (drdf).pow(2).mean()#+ JS_rdf(g_sim, rdf_target_list[j])

                self.obs_log[data_str]['rdf'].append(g_sim.detach().cpu().numpy())
                self.obs_log[data_str]['vacf'].append(vacf_sim.detach().cpu().numpy())

                if i % 5 ==0 :
                    if self.vacf_target_list[j] is not None:
                        vacf_target = self.vacf_target_list[j][:self.t_range].detach().cpu().numpy()
                    else:
                        vacf_target = None
                    rdf_target = self.rdf_target_list[j].detach().cpu().numpy()

                    plot_vacf(vacf_sim.detach().cpu().numpy(), vacf_target, 
                        fn=data_str + "_{}".format(str(i).zfill(3)), 
                        dt=dt,
                        path=self.model_path)

                    plot_rdf(g_sim.detach().cpu().numpy(), rdf_target, 
                        fn=data_str + "_{}".format(str(i).zfill(3)),
                            path=self.model_path, 
                            start=self.rdf_start, 
                            nbins=self.nbins,
                            end=self.rdf_obs_list[j].r_axis[-1])

                if i % 5 ==0 :
                    print('qui')
                    print("Model Structure:", self.sim.integrator.model)

                    potential = plot_pair( path=self.model_path,
                                    fn=str(i).zfill(3),
                                    model=self.sim.integrator.model.models['pairnn'].model, 
                                    prior=self.sim.integrator.model.models['pair'].model, 
                                    device=device,
                                    target_pot=target_pot.to(device),
                                    end=cutoff)
                    print('ora qui oki')

            if self.train_vacf == "True":
                loss = self.rdf_weight * loss_rdf +  self.vacf_weight * loss_vacf
            else:
                loss = self.rdf_weight * loss_rdf
            print('back start')
            loss.backward() # now we call the gambero back back 

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            print(loss_vacf.item(), loss_rdf.item())
            
            self.scheduler.step(loss)
            
            self.loss_log.append([loss_vacf.item(), loss_rdf.item() ])

            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= 1e-5:
                print("training converged")
                break

            np.savetxt(self.model_path + '/loss.txt', np.array(self.loss_log), delimiter=',')
        print('================= DONE =================')
        # # save potentials         
        # if np.array(loss_log)[-10:, 1].mean() <=  0.005: 
        #     np.savetxt(model_path + '/potential.txt',  potential, delimiter=',')

        rdf_dev = []
        n_sim = 10
        for j, sim in enumerate(sim_list):
            print('sim sim: ', j)
            dt = 0.005

            all_vacf_sim = []

            for i in range(n_sim):
                print('entering this other for: ',i)
                    
                if selected_integrator == "NoseHooverChain":
                    #print('here')
                    v_t, q_t, pv_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.
                elif selected_integrator == "AndersenODE":
                    #print('here now')
                    v_t, q_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.


                # compute VACF 
                vacf_sim = self.vacf_obs_list[j](v_t).detach().cpu().numpy()
                all_vacf_sim.append(vacf_sim)

            all_vacf_sim = np.array(all_vacf_sim).mean(0)
            
            trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device).detach()

            # get targets
            if self.vacf_target_list[j] is not None:
                vacf_target = self.vacf_target_list[j][:self.t_range].detach().cpu().numpy()
            else:
                vacf_target = None
            rdf_target = self.rdf_target_list[j].detach().cpu().numpy()
            

            # loop over to ocmpute observables 
            print('1----')
            all_g_sim = []
            for i in range(len(trajs)):
                _, _, g_sim = self.rdf_obs_list[j](trajs[[i]])
                all_g_sim.append(g_sim.detach().cpu().numpy())

            all_g_sim = np.array(all_g_sim).mean(0)
            
            # compute target deviation 
            if data_str in self.main_app.data_str_list:
                drdf = np.abs(all_g_sim - self.rdf_target_list[j].cpu().numpy()).mean()
                rdf_dev.append(drdf) 

            # plot observables 
            plot_vacf(all_vacf_sim, vacf_target, 
                fn=data_str, 
                path=self.model_path,
                dt=dt,
                save_data=True)

            plot_rdf(all_g_sim, rdf_target, 
                fn=data_str,
                    path=self.model_path, 
                    start=self.rdf_start, 
                    nbins=self.nbins,
                    save_data=True,
                    end=self.rdf_obs_list[j].r_axis[-1])

        print('======================= DONE =======================')
            # rdf_dev = np.abs(all_g_sim - rdf_target).mean()

        # Save the learned potential if the final RDF loss is below threshold
        if np.array(self.loss_log)[-10:, 1].mean() <= 0.005: 
            potential = plot_pair(path=self.model_path,
                                fn="final",
                                model=self.sim.integrator.model.models['pairnn'].model, 
                                prior=self.sim.integrator.model.models['pair'].model, 
                                device=device,
                                target_pot=target_pot.to(device),
                                end=cutoff)
            
            np.savetxt(self.model_path + '/potential.txt', potential, delimiter=',')

        rdf_dev = []
        import ase
        import ase.io

        # Function to save trajectory as XYZ file
        def save_traj(system, traj, fname, skip=10):
            """
            Saves atomic trajectories as an XYZ file for visualization.

            Parameters:
                system: ASE Atoms object representing the molecular system.
                traj: List of atomic positions at different timesteps.
                fname: Filename for saving the trajectory.
                skip: Interval for saving frames (reduces file size).
            """
            atoms_list = []
            for i, frame in enumerate(traj):
                if i % skip == 0:  # Save every 'skip' frames to reduce file size
                    frame = ase.Atoms(positions=frame, numbers=system.get_atomic_numbers())
                    atoms_list.append(frame)
            ase.io.write(fname, atoms_list)  # Save as XYZ file

        # Final evaluation loop
        for j, sim in enumerate(sim_list):
            print('======================= EVAL MPODE =======================')
            # Simulate with the trained model (without optimization)
            train_traj = sim.log['positions']
            print(i)
            # Save training trajectory
            save_traj(self.main_app.system_list[j], train_traj, self.model_path + '/{}_train.xyz'.format(data_str), skip=10)

            dt = 0.005
            all_vacf_sim = []

            for i in range(n_sim):
                if selected_integrator == "NoseHooverChain":
                    #print('here')
                    v_t, q_t, pv_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.
                elif selected_integrator == "AndersenODE":
                    #print('here now')
                    v_t, q_t = self.sim.simulate(100, dt=dt, frequency=100) # 100 time steps, dt is given with argparse are in fs, freq is how often the system's state is updated and logged during a md simulation.


                # Compute VACF
                vacf_sim = self.vacf_obs_list[j](v_t).detach().cpu().numpy()
                all_vacf_sim.append(vacf_sim)

            all_vacf_sim = np.array(all_vacf_sim).mean(0)
            
            trajs = torch.Tensor(np.stack(sim.log['positions'])).to(system.device).detach()

            # Get target RDF and VACF
            if self.vacf_target_list[j] is not None:
                vacf_target = self.vacf_target_list[j][:self.t_range].detach().cpu().numpy()
            else:
                vacf_target = None
            rdf_target = self.rdf_target_list[j].detach().cpu().numpy()
            
            # Compute RDF over all saved trajectories
            all_g_sim = []
            for i in range(len(trajs)):
                _, _, g_sim = self.rdf_obs_list[j](trajs[[i]])
                all_g_sim.append(g_sim.detach().cpu().numpy())

            all_g_sim = np.array(all_g_sim).mean(0)
            
            # Compute target deviation
            if data_str in self.main_app.data_str_list:
                drdf = np.abs(all_g_sim - self.rdf_target_list[j].cpu().numpy()).mean()
                rdf_dev.append(drdf) 

            # Plot and save observables (VACF and RDF)
            plot_vacf(all_vacf_sim, vacf_target, 
                    fn=data_str, 
                    path=self.model_path,
                    dt=dt,
                    save_data=True)

            plot_rdf(all_g_sim, rdf_target, 
                    fn=data_str,
                    path=self.model_path, 
                    start=self.rdf_start, 
                    nbins=self.nbins,
                    save_data=True,
                    end=self.rdf_obs_list[j].r_axis[-1])

            # Save final trajectory (inference trajectory)
            save_traj(self.main_app.system_list[j], np.stack(sim.log['positions']),  
                    self.model_path + '/{}_sim.xyz'.format(data_str), skip=1)
        print('EVERYTHING DONE :D')