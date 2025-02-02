import tkinter as tk
from GUI import GUI
from tkinter import messagebox
from ase.visualize import view
import torch

from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *

from simulationGUI import SimulationGUI


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



class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Main GUI")
        
        self.app = None  # To store the GUI instance
        self.system_list = []  # Store generated systems

        self.model_list = None
        self.NN = None
        self.prior = None

        # Buttons
        self.open_gui_btn = tk.Button(root, text="Open Data Input GUI", command=self.open_gui)
        self.open_gui_btn.pack(pady=10)

        self.load_systems_btn = tk.Button(root, text="Load Systems", command=self.load_systems, state=tk.DISABLED)
        self.load_systems_btn.pack(pady=10)

        self.view_systems_btn = tk.Button(root, text="View Systems", command=self.view_systems, state=tk.DISABLED)
        self.view_systems_btn.pack(pady=10)

        self.load_models_btn = tk.Button(root, text="Load Models", command=self.load_models)
        self.load_models_btn.pack(pady=10)

        self.run_simulation_btn = tk.Button(root, text="Open Simulation GUI", command=self.open_simulation)
        self.run_simulation_btn.pack(pady=10)

        # Output Text Box
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.pack(pady=10)

    def open_gui(self):
        """Opens the existing Pair Data Input GUI."""
        new_window = tk.Toplevel(self.root)
        self.app = GUI(new_window)  # Open GUI in a new window
        self.load_systems_btn.config(state=tk.NORMAL)  # Enable system loading

    def open_simulation(self):
        """Opens the Simulation GUI."""
        if not self.model_list:
            messagebox.showwarning("Warning", "No models loaded! Load models first.")
            return
        
        new_window = tk.Toplevel(self.root)
        SimulationGUI(new_window, self)  # Pass MainGUI instance

    def load_systems(self):
        """Loads system data from the GUI."""
        if self.app is None:
            messagebox.showerror("Error", "Please open the GUI first!")
            return
        
        self.system_list = self.app.system_list  # Retrieve generated systems
        self.data_str_list = self.app.data_str_list
        self.pair_data_dict = self.app.pair_data_dict

        if not self.system_list:
            messagebox.showwarning("Warning", "No systems generated yet!")
        else:
            messagebox.showinfo("Success", f"Loaded {len(self.system_list)} systems!")
            self.view_systems_btn.config(state=tk.NORMAL)  # Enable View Systems button

    def view_systems(self):
        """Displays loaded systems in the output box."""
        self.output_text.delete("1.0", tk.END)
        view(self.system_list[0])
        if not self.system_list:
            self.output_text.insert(tk.END, "No systems loaded.\n")
        else:
            self.output_text.insert(tk.END, f"Loaded {len(self.system_list)} systems:\n")
            for i, system in enumerate(self.system_list):
                self.output_text.insert(tk.END, f"System {i+1}: {system}\n")
            
    def load_models(self):
        """Generate and store machine learning models for the loaded systems."""
        if not self.system_list:
            messagebox.showerror("Error", "No systems loaded! Load systems first.")
            return
        nbr_list_device='cpu'
        device='cpu'
        cutoff = 2.5
        mlp_params = {
            'n_gauss': int(cutoff // 0.10),  # cutoff // gaussian_width 
            'r_start': 0.0,
            'r_end': cutoff, 
            'n_width': 128,
            'n_layers': 3,
            'nonlinear': 'ELU'
        }
        lj_params = {
            'epsilon': 0.4, 
            'sigma': 0.9,
            "power": 10
        }

        self.NN = pairMLP(**mlp_params).to(device)
        self.prior = LJFamily(epsilon=2.0, sigma=0.9, rep_pow=6, attr_pow=3).to(device)

        self.model_list = [] 
        for i, data_str in enumerate(self.data_str_list):
            system = self.system_list[i]

            pairNN = PairPotentials(system, self.NN,
                                    cutoff=cutoff,
                                    nbr_list_device=nbr_list_device).to(device)
            
            prior_model = PairPotentials(system, self.prior,
                                        cutoff=2.5,
                                        nbr_list_device=nbr_list_device).to(device)

            model = Stack({'pairnn': pairNN, 'pair': prior_model}).to(device)
          
            self.model_list.append(model)

        messagebox.showinfo("Success", f"{len(self.model_list)} models generated!")
