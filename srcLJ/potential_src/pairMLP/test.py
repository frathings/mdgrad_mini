from potential_PairMLP import *
from torchviz import make_dot

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
        'data': 'Si_2.293_100K',
        'val': None,
        'size': 4,
        'anneal_flag': True,
        'pair_flag': True,
    }

net, prior = get_pair_potential(assignments, sys_params)
print('NET PRIOR _______________________')
print(net)
print('NET PRIOR _______________________')
print(prior)
print('NET PRIOR _______________________')

import os

# Append Graphviz's bin directory to PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Example input tensor
dummy_input = torch.randn(1, 33)  # Adjust the dimensions to match your model's expected input

# Generate the output from the model
output = net(dummy_input)

# Create a graph of the model
dot = make_dot(output, params=dict(net.named_parameters()))

# Save the graph to a file
dot.format = 'png'  # or 'pdf', 'svg', etc.
dot.render('model_graph')
