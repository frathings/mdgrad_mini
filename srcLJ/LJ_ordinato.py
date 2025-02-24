import os
import json
import numpy as np
import matplotlib.pyplot as plt

import questionary
from rich import print, box
from rich.console import Console
from rich.table import Table

from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic, Diamond
from ase.visualize import *

from data_src.data import *
from potential_src.pairMLP.potential_PairMLP import *
from observables.rdf import *
from observables.observers import *   
from utils.get_utils import *


# Definizione dei dati di default
pair_data_dict = {
    'lj_0.3_1.2': {
        'rho': 0.3,
        'T': 1.2,
        'start': 0.75,
        'end': 3.3,
        'element': "H",
        'mass': 1.0,
        "N_unitcell": 4,
        "cell": FaceCenteredCubic,
        "target_pot": LennardJones(sigma=1, epsilon=1)
    }
}

# Funzione per mostrare i dati
console = Console()
def show_data(data):
    table = Table(title="Dati di Simulazione", box=box.ROUNDED)
    table.add_column("Parametro", style="cyan", justify="left")
    table.add_column("Valore", style="magenta", justify="right")
    
    for key, value in data.items():
        table.add_row(key, str(value))
    
    console.print(table)

# Funzione per richiedere input all'utente
def get_user_input():
    pair_data_dict['lj_0.3_1.2']['rho'] = float(questionary.text("Inserisci la densità (rho):", default=str(pair_data_dict['lj_0.3_1.2']['rho'])).ask())
    pair_data_dict['lj_0.3_1.2']['T'] = float(questionary.text("Inserisci la temperatura (T):", default=str(pair_data_dict['lj_0.3_1.2']['T'])).ask())
    pair_data_dict['lj_0.3_1.2']['element'] = questionary.text("Inserisci l'elemento chimico:", default=pair_data_dict['lj_0.3_1.2']['element']).ask()
    pair_data_dict['lj_0.3_1.2']['start'] = float(questionary.text("Inserisci il valore di inizio:", default=str(pair_data_dict['lj_0.3_1.2']['start'])).ask())
    pair_data_dict['lj_0.3_1.2']['end'] = float(questionary.text("Inserisci il valore di fine:", default=str(pair_data_dict['lj_0.3_1.2']['end'])).ask())
    
    return pair_data_dict['lj_0.3_1.2']





















if __name__ == "__main__":
    sim_data = get_user_input()
    show_data(sim_data)