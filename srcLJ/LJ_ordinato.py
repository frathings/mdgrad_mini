from LJ_ordinato_func import *

topology_update_freq = 1

###################
# Get user input
###################

sim_data = get_user_input()
update_config(sim_data, system_key='lj_0.3_1.2')

# Display system parameters
# display_data(CONFIG["systems"]["lj_0.3_1.2"], title="System Configuration")
# Display RDF parameters
# display_data(CONFIG["observables"]["rdf"], title="RDF Configuration")
# Display combined data
display_data({
    **CONFIG["systems"]["lj_0.3_1.2"],
    "r_range": CONFIG["observables"]["rdf"]["r_range"]
}, title="Combined Configuration")

###################
# System
###################

data_str_list = ['lj_0.3_1.2']
params = {'val': []}
sys_params = {'val': params['val']}

if sys_params['val']:
    val_str_list = sys_params['val']
else:
    val_str_list = []
system_list = []
for data_str in data_str_list+val_str_list:
    system = get_system(data_str, CONFIG["simulation"]["device"], CONFIG["simulation"]["size"]) 
    system_list.append(system)

from ase.visualize import view
# view(system_list[0])

###################
# Potential
###################

# Create models based on user selection
models = create_models(
    system_list=system_list,
    data_str_list=data_str_list,
    val_str_list=val_str_list,
    device=CONFIG["simulation"]["device"],
    cutoff=CONFIG["systems"]["lj_0.3_1.2"]["cutoff"]
)
# For each model
for i, model in enumerate(models):
    console.print(f"\n[bold cyan]Model {i+1}:[/bold cyan]")
    count_parameters(model)

###################
# Simulation
###################

sim_list = [get_sim(system_list[i], 
                    models[i], 
                    data_str,
                    topology_update_freq=topology_update_freq) for i, data_str in enumerate(data_str_list + val_str_list)]
print(sim_list)


from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
import os
import numpy as np
import torch
import pickle

# Create a directory for saved data
data_dir = "./saved_data"
os.makedirs(data_dir, exist_ok=True)

# Define a function to save observer data
def save_observer_data(rdf_bins_list, rdf_target_list, vacf_target_list, system_keys):
    data = {
        'rdf_bins_list': [x.numpy() if torch.is_tensor(x) else x for x in rdf_bins_list],
        'rdf_target_list': [rt.cpu().numpy() if torch.is_tensor(rt) else rt for rt in rdf_target_list],
        'vacf_target_list': [vt.cpu().numpy() if torch.is_tensor(vt) and vt is not None else vt for vt in vacf_target_list],
        'system_keys': system_keys
    }
    
    with open(f"{data_dir}/observer_data.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    console.print(f"[green]✓[/green] Observer data saved to {data_dir}/observer_data.pkl")

# Define a function to load observer data
def load_observer_data(system_list, data_str_list, val_str_list, device, nbins, t_range):
    system_keys = data_str_list + val_str_list
    
    # Check if saved data exists
    if os.path.exists(f"{data_dir}/observer_data.pkl"):
        # Ask if user wants to use saved data
        use_saved = questionary.confirm(
            "Found saved observer data. Do you want to use it?", 
            default=True
        ).ask()
        
        if use_saved:
            try:
                with open(f"{data_dir}/observer_data.pkl", 'rb') as f:
                    data = pickle.load(f)
                
                # Check if the saved data matches current systems
                if data['system_keys'] == system_keys:
                    console.print("[bold green]Loading saved observer data...[/bold green]")
                    
                    # Convert back to tensors where needed
                    rdf_bins_list = data['rdf_bins_list']
                    
                    # Create new observer objects with loaded target data
                    rdf_obs_list = []
                    vacf_obs_list = []
                    rdf_target_list = []
                    vacf_target_list = []
                    
                    for i, system in enumerate(system_list):
                        # Get rdf_start from bin data
                        rdf_start = data['rdf_bins_list'][i][0]
                        rdf_end = data['rdf_bins_list'][i][-1]
                        
                        # Create new observers
                        rdf_obs = rdf(system, nbins=nbins, r_range=(rdf_start, rdf_end))
                        vacf_obs = vacf(system, t_range=t_range)
                        
                        # Convert target data back to tensors
                        rdf_target = torch.tensor(data['rdf_target_list'][i], device=device)
                        
                        # Handle VACF target (which might be None)
                        if data['vacf_target_list'][i] is not None:
                            vacf_target = torch.tensor(data['vacf_target_list'][i], device=device)
                        else:
                            vacf_target = None
                        
                        # Store in lists
                        rdf_obs_list.append(rdf_obs)
                        vacf_obs_list.append(vacf_obs)
                        rdf_target_list.append(rdf_target)
                        vacf_target_list.append(vacf_target)
                    
                    console.print(f"[green]✓[/green] Successfully loaded observer data for {len(rdf_obs_list)} systems")
                    return rdf_bins_list, rdf_target_list, rdf_obs_list, vacf_target_list, vacf_obs_list
                else:
                    console.print("[yellow]Saved data doesn't match current systems. Computing new data...[/yellow]")
            except Exception as e:
                console.print(f"[red]Error loading saved data: {e}[/red]")
                console.print("[yellow]Computing new observer data...[/yellow]")
    else:
        console.print("[yellow]No saved data found. Computing new observer data...[/yellow]")
    
    # If we get here, we need to compute new data
    return None

# Now modify your observer setup code to use these functions
rdf_obs_list = []
vacf_obs_list = []
rdf_target_list = []
vacf_target_list = []
rdf_bins_list = []

console.print("\n[bold]Setting up observers for all systems...[/bold]")

# Try to load saved data first
loaded_data = load_observer_data(
    system_list, 
    data_str_list, 
    val_str_list, 
    CONFIG["simulation"]["device"], 
    CONFIG["observables"]["rdf"]["nbins"], 
    CONFIG["simulation"]["t_range"]
)

if loaded_data:
    # Unpack loaded data
    rdf_bins_list, rdf_target_list, rdf_obs_list, vacf_target_list, vacf_obs_list = loaded_data
else:
    # Compute new data with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        # Create a task for setting up observers
        obs_task = progress.add_task(
            "[bold]Setting up observers...", 
            total=len(data_str_list + val_str_list)
        )
        
        for i, data_str in enumerate(data_str_list + val_str_list):
            # Get rdf_start from config
            rdf_start = CONFIG["observables"]["rdf"]["r_range"]["start"]
            
            # Update task description
            progress.update(
                obs_task, 
                description=f"[bold]Setting up observer for {data_str} ({i+1}/{len(data_str_list + val_str_list)})"
            )
            
            x, rdf_target, rdf_obs, vacf_target, vacf_obs = get_observer(
                system=system_list[i],
                system_key=data_str, 
                nbins=CONFIG["observables"]["rdf"]["nbins"], 
                t_range=CONFIG["simulation"]["t_range"],
                rdf_start=rdf_start
            )
            
            # Store results in lists
            rdf_bins_list.append(x)
            rdf_obs_list.append(rdf_obs)
            rdf_target_list.append(rdf_target)
            vacf_obs_list.append(vacf_obs)
            vacf_target_list.append(vacf_target)
            
            # Advance the progress bar
            progress.update(obs_task, advance=1)
        
        # Save the computed data for future use
        save_observer_data(rdf_bins_list, rdf_target_list, vacf_target_list, data_str_list + val_str_list)

console.print(f"[green]✓[/green] Set up observers for {len(rdf_obs_list)} systems")


# After setting up observers (either loaded or computed)
console.print("\n[bold]Plotting RDF targets...[/bold]")
# Import matplotlib if not already imported
import matplotlib.pyplot as plt
# Create a figure for RDF targets
plt.figure(figsize=(10, 6))
# Plot each RDF target
for i, data_str in enumerate(data_str_list + val_str_list):
    # Convert tensor to numpy if needed
    if torch.is_tensor(rdf_target_list[i]):
        rdf_target = rdf_target_list[i].cpu().numpy()
    else:
        rdf_target = rdf_target_list[i]
    
    # Plot with different colors for each system
    plt.plot(
        rdf_bins_list[i], 
        rdf_target, 
        label=f"{data_str}", 
        linewidth=2
    )
# Add labels and title
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function (RDF) Targets")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# Display the plot
plt.show()

############################
# Training Iniitalization
############################

# After setting up models
optimizer, scheduler, model_type, loss_log, obs_log = setup_optimizer_and_scheduler(models)

for i, data_str in enumerate(data_str_list + val_str_list):
    obs_log[data_str] = {}
    obs_log[data_str]['rdf'] = []
    obs_log[data_str]['vacf'] = []
rdf_weight = 1
vacf_weight = 0.0


# Train the model
loss_log, obs_log = train_model(
    models=models,
    sim_list=sim_list,
    rdf_obs_list=rdf_obs_list,
    vacf_obs_list=vacf_obs_list,
    rdf_target_list=rdf_target_list,
    vacf_target_list=vacf_target_list,
    rdf_bins_list=rdf_bins_list,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cpu',
    n_epochs=1000,
    model_path="./results",
    rdf_weight=1.0,
    vacf_weight=0.1,
    train_vacf="True",
    cutoff=2.5,
    target_pot=CONFIG["systems"]["lj_0.3_1.2"]["target_pot"]
)

