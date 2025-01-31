from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.visualize import view
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt

def create_structure(structure_type="fcc", element="Cu", size=(3, 3, 3)):
    """
    Create a crystal structure.

    Parameters:
    - structure_type (str): Type of structure to create. Default is 'fcc'.
    - element (str): Element symbol (e.g., 'Cu'). Default is 'Cu'.
    - size (tuple): Size of the structure (e.g., (3, 3, 3)). Default is (3, 3, 3).

    Returns:
    - Atoms: ASE Atoms object representing the crystal.
    """
    if structure_type.lower() == "fcc":
        return FaceCenteredCubic(element, size=size)
    else:
        raise ValueError(f"Unsupported structure type: {structure_type}")


def attach_calculator(atoms, calculator=None):
    """
    Attach a calculator to the Atoms object.

    Parameters:
    - atoms (Atoms): ASE Atoms object.
    - calculator (Calculator): Calculator to attach. Default is EMT.

    Returns:
    - None
    """
    if calculator is None:
        calculator = EMT()
    atoms.calc = calculator


def initialize_velocities(atoms, temperature_K=300):
    """
    Initialize velocities based on the Maxwell-Boltzmann distribution.

    Parameters:
    - atoms (Atoms): ASE Atoms object.
    - temperature_K (float): Temperature in Kelvin. Default is 300 K.

    Returns:
    - None
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)


def print_energy(atoms, step):
    """
    Print potential, kinetic, and total energy per atom.

    Parameters:
    - atoms (Atoms): ASE Atoms object.
    - step (int): Current simulation step.

    Returns:
    - None
    """
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    etot = epot + ekin
    print(f"Step {step}, Epot = {epot:.3f}eV, Ekin = {ekin:.3f}eV, Etot = {etot:.3f}eV")


def run_md_simulation(
    atoms, steps=20, timestep_fs=5, energy_print_interval=10, temperature_K=300
):
    """
    Run a molecular dynamics simulation and return energy data.

    Parameters:
    - atoms (Atoms): ASE Atoms object.
    - steps (int): Total number of steps to simulate. Default is 20.
    - timestep_fs (float): Timestep in femtoseconds. Default is 5 fs.
    - energy_print_interval (int): Interval for printing energy data. Default is 10 steps.
    - temperature_K (float): Temperature in Kelvin for initial velocities. Default is 300 K.

    Returns:
    - dict: A dictionary containing lists of potential, kinetic, and total energies.
    """
    initialize_velocities(atoms, temperature_K=temperature_K)
    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    energies = {"potential": [], "kinetic": [], "total": []}

    def record_energy():
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        etot = epot + ekin
        energies["potential"].append(epot)
        energies["kinetic"].append(ekin)
        energies["total"].append(etot)

    # Record initial energy
    record_energy()

    for step in range(1, steps + 1):
        dyn.run(1)
        record_energy()
        if step % energy_print_interval == 0:
            print(
                f"Step {step}, "
                f"Epot = {energies['potential'][-1]:.3f}eV, "
                f"Ekin = {energies['kinetic'][-1]:.3f}eV, "
                f"Etot = {energies['total'][-1]:.3f}eV"
            )

    return energies


def plot_energy(energies, timestep_fs):
    """
    Plot potential, kinetic, and total energy over time.

    Parameters:
    - energies (dict): Dictionary containing 'potential', 'kinetic', and 'total' energy lists.
    - timestep_fs (float): Timestep in femtoseconds.

    Returns:
    - None
    """
    timesteps = [i * timestep_fs for i in range(len(energies["potential"]))]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, energies["potential"], label="Potential Energy (eV)", linestyle='-', marker='o')
    plt.plot(timesteps, energies["kinetic"], label="Kinetic Energy (eV)", linestyle='-', marker='x')
    plt.plot(timesteps, energies["total"], label="Total Energy (eV)", linestyle='-', marker='s')
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title("Energy Evolution During Molecular Dynamics")
    plt.legend()
    plt.grid()
    plt.show()


def draw_system(atoms):
    """
    Visualize the atomic system in an interactive viewer.

    Parameters:
    - atoms (Atoms): ASE Atoms object representing the system.

    Returns:
    - None
    """
    view(atoms)


def plot_3d_system(atoms):
    """
    Plot a 3D representation of the atomic system using Matplotlib.

    Parameters:
    - atoms (Atoms): ASE Atoms object representing the system.

    Returns:
    - None
    """
    positions = atoms.get_positions()
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', s=100, alpha=0.7)
    ax.set_xlabel("X-axis (Å)")
    ax.set_ylabel("Y-axis (Å)")
    ax.set_zlabel("Z-axis (Å)")
    ax.set_title("3D Atomic Structure")
    plt.show()
    
def calculate_rdf(atoms, r_max=10.0, nbins=100):
    """
    Calculate the Radial Distribution Function (RDF) using MDAnalysis.

    Parameters:
    - atoms (ASE Atoms): ASE Atoms object representing the system.
    - r_max (float): Maximum distance (Å) for RDF calculation. Default is 10.0 Å.
    - nbins (int): Number of bins for the histogram. Default is 100.

    Returns:
    - r (np.ndarray): Array of distances (Å).
    - g_r (np.ndarray): RDF values at corresponding distances.
    """
    # Step 1: Create an empty universe
    n_atoms = len(atoms)
    u = mda.Universe.empty(
        n_atoms, trajectory=True
    )  # Empty universe with `n_atoms`

    # Step 2: Add positions from ASE Atoms
    u.atoms.positions = atoms.get_positions()

    # Step 3: Add topology (element symbols)
    u.add_TopologyAttr("element", atoms.get_chemical_symbols())

    # Step 4: Set the simulation box dimensions
    u.dimensions = atoms.get_cell_lengths_and_angles()

    # Step 5: Define atom groups for RDF calculation
    group1 = u.atoms
    group2 = u.atoms

    # Step 6: Perform RDF calculation
    rdf = InterRDF(group1, group2, nbins=nbins, range=(0.0, r_max))
    rdf.run()

    return rdf.bins, rdf.rdf


def plot_rdf(r, g_r):
    """
    Plot the Radial Distribution Function (RDF).

    Parameters:
    - r (np.ndarray): Array of distances (Å).
    - g_r (np.ndarray): RDF values at corresponding distances.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(r, g_r, label="Radial Distribution Function (RDF)")
    plt.xlabel("Distance (Å)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.grid(True)
    plt.legend()
    plt.show()
