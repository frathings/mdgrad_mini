
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic, Diamond

from ase.visualize import *
from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *

predefined_data = {
    'lj_0.3_1.2': {
        'rdf_fn': '../data/LJ_data/rdf_rho0.3_T1.2_dt0.01.csv',
        'vacf_fn': '../data/LJ_data/vacf_rho0.3_T1.2_dt0.01.csv',
        'rho': 0.3,
        'T': 1.2,
        'start': 0.75,
        'end': 3.3,
        'element': "H",
        'mass': 1.0,
        "N_unitcell": 4,
        "cell": "FaceCenteredCubic",
        "target_pot": LennardJones()
    }
}

def get_system(data_str, device, size, pair_data_dict):
    """Initialize an NVT system based on pair_data_dict."""
    rho = pair_data_dict[data_str]['rho']
    T = pair_data_dict[data_str]['T']
    dim = pair_data_dict[data_str].get("dim", 3)

    if dim == 3:
        cell_module = pair_data_dict[data_str]['cell']
        N_unitcell = pair_data_dict[data_str]['N_unitcell']

        def get_unit_len(rho, N_unitcell):
            return (N_unitcell / rho) ** (1/3)

        L = get_unit_len(rho, N_unitcell)
        print("Lattice parameter:", L)

        atoms = cell_module(symbol=pair_data_dict[data_str]['element'],
                            size=(size, size, size),
                            latticeconstant=L,
                            pbc=True)

        system = System(atoms, device=device)
        system.set_temperature(T)
        return system

    return None  # Return None if system creation fails


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pair Data Input")
        
        self.pair_data_dict = predefined_data.copy()  # Load predefined data
        self.create_widgets()
        self.load_predefined_data()
        self.system_list = []
        self.data_str_list = list()

    def create_widgets(self):
        # Labels and Entry Fields
        labels = ["Density (rho):", "Temperature (T):", "Start:", "End:", "Element:", "Mass:", "N_unitcell:"]
        self.entries = {}
        
        for i, text in enumerate(labels):
            tk.Label(self.root, text=text).grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.root)
            entry.grid(row=i, column=1)
            self.entries[text] = entry
            
        # File Selection Buttons
        self.file_vars = {"rdf_fn": tk.StringVar(), "vacf_fn": tk.StringVar()}
        
        for i, (key, var) in enumerate(self.file_vars.items(), start=len(labels)):
            tk.Label(self.root, text=f"{key}:").grid(row=i, column=0, sticky='w')
            entry = tk.Entry(self.root, textvariable=var)
            entry.grid(row=i, column=1)
            btn = tk.Button(self.root, text="Browse", command=lambda v=var: self.browse_file(v))
            btn.grid(row=i, column=2)
            
        # Dropdowns for cell and potential
        self.cell_var = tk.StringVar(value="FaceCenteredCubic")
        self.pot_var = tk.StringVar(value="LennardJones")
        
        tk.Label(self.root, text="Cell Type:").grid(row=i+1, column=0, sticky='w')
        ttk.Combobox(self.root, textvariable=self.cell_var, values=["FaceCenteredCubic"]).grid(row=i+1, column=1)
        
        tk.Label(self.root, text="Target Potential:").grid(row=i+2, column=0, sticky='w')
        ttk.Combobox(self.root, textvariable=self.pot_var, values=["LennardJones"]).grid(row=i+2, column=1)
        
        # Save Button
        tk.Button(self.root, text="Save Data", command=self.save_data).grid(row=i+3, column=0, columnspan=2)

        # FIX: Use a new row for "Generate Systems" button
        tk.Button(self.root, text="Generate Systems", command=self.generate_systems).grid(row=i+4, column=0, columnspan=2)

        # FIX: Move text output to the next row (i+5)
        self.output_text = tk.Text(self.root, height=10, width=50)
        self.output_text.grid(row=i+5, column=0, columnspan=2)

    def generate_systems(self):
        """Generate system instances and display results."""
        self.system_list = []  # Clear existing system list
        #data_str_list = list(self.pair_data_dict.keys())  # Get all saved keys
        self.data_str_list = list(self.pair_data_dict.keys())
        if not self.data_str_list:
            messagebox.showwarning("Warning", "No data saved to generate systems!")
            return

        for data_str in self.data_str_list:
            system = get_system(data_str, "cpu", self.pair_data_dict[data_str]["N_unitcell"], self.pair_data_dict)
            if system:
                self.system_list.append(system)

        # Display results in the GUI
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Generated {len(self.system_list)} systems:\n")
        self.output_text.insert(tk.END, "\n".join(self.data_str_list))

        messagebox.showinfo("Success", f"{len(self.system_list)} systems generated!")

    def load_predefined_data(self):
        for key, data in self.pair_data_dict.items():
            self.entries["Density (rho):"].insert(0, str(data['rho']))
            self.entries["Temperature (T):"].insert(0, str(data['T']))
            self.entries["Start:"].insert(0, str(data['start']))
            self.entries["End:"].insert(0, str(data['end']))
            self.entries["Element:"].insert(0, data['element'])
            self.entries["Mass:"].insert(0, str(data['mass']))
            self.entries["N_unitcell:"].insert(0, str(data['N_unitcell']))
            self.file_vars["rdf_fn"].set(data['rdf_fn'])
            self.file_vars["vacf_fn"].set(data['vacf_fn'])
            self.cell_var.set(data['cell'])
            self.pot_var.set(data['target_pot'])
            break  # Load only the first available entry
    
    def browse_file(self, var):
        filename = filedialog.askopenfilename()
        var.set(filename)
    
    def save_data(self):
        key = f"lj_{self.entries['Density (rho):'].get()}_{self.entries['Temperature (T):'].get()}"
        self.pair_data_dict = {}  # Clear predefined data

        self.pair_data_dict[key] = {
            'rdf_fn': self.file_vars['rdf_fn'].get(),
            'vacf_fn': self.file_vars['vacf_fn'].get(),
            'rho': float(self.entries['Density (rho):'].get()),
            'T': float(self.entries['Temperature (T):'].get()),
            'start': float(self.entries['Start:'].get()),
            'end': float(self.entries['End:'].get()),
            'element': self.entries['Element:'].get(),
            'mass': float(self.entries['Mass:'].get()),
            'N_unitcell': int(self.entries['N_unitcell:'].get()),
            'cell': FaceCenteredCubic if self.cell_var.get() == "FaceCenteredCubic" else None,  # ✅ Store as class
            'target_pot': self.pot_var.get(),
        }
        
        # Display updated dictionary
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, str(self.pair_data_dict))
        
        messagebox.showinfo("Success", "Data Saved Successfully")
