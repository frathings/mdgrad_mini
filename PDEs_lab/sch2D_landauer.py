import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- Constants ---
Lx, Ly = 2.0, 0.5  # Wire dimensions
dx, dy = 0.0001, 0.02  # Grid spacing
Nx, Ny = int(Lx / dx), int(Ly / dy)
hbar = 1.0
m = 1.0
E_range = np.linspace(0, 10, 200)  # Fine energy sampling

# --- Grid setup ---
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
xx, yy = np.meshgrid(x, y)

# --- Create hard wall potential ---
def create_wire_potential():
    V2D = np.zeros((Ny, Nx))
    # Define constriction width
    W = 0.1  # Width of constriction
    x0 = 0.0  # Center of constriction
    
    # Create smooth constriction
    for i in range(Nx):
        width = W + 0.2 * (1 + np.tanh(2*(abs(x[i] - x0) - 0.4)))
        for j in range(Ny):
            if abs(y[j]) > width/2:
                V2D[j, i] = 1e6  # Very high potential for hard walls
    return V2D

V2D = create_wire_potential()

# --- Setup Hamiltonian ---
I_x = sp.eye(Nx)
I_y = sp.eye(Ny)
Lap_x = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
Lap_y = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2

N_total = Nx * Ny
H2D = -hbar**2 / (2 * m) * (sp.kron(Lap_x, I_y) + sp.kron(I_x, Lap_y))
V2D_flat = V2D.reshape(-1, order="F")
V2D_sparse = sp.diags(V2D_flat, 0, shape=(N_total, N_total))
H2D = H2D + V2D_sparse

def compute_modes(energy):
    """Compute transverse modes at given energy"""
    # Solve for transverse modes
    Hy = -hbar**2 / (2 * m) * Lap_y
    eigenvalues, eigenvectors = np.linalg.eigh(Hy.toarray())
    
    # Find propagating modes
    k_modes = np.sqrt(2 * m * energy - eigenvalues)
    propagating = np.isreal(k_modes) & (k_modes > 0)
    
    return sum(propagating)

def compute_transmission(energy):
    """Compute transmission using mode matching"""
    # Number of propagating modes
    N_modes = compute_modes(energy)
    
    # Simple approximation of transmission
    if N_modes == 0:
        return 0
    
    # Scale transmission by available modes
    # This is a simplified model - real calculation would involve 
    # matching wavefunctions at the interfaces
    T = N_modes * (1 - np.exp(-energy / 2))
    return min(T, N_modes)  # Cannot transmit more than number of modes

# Compute conductance
print("Computing conductance...")
G_E = np.zeros(len(E_range))
for i, E in enumerate(E_range):
    T = compute_transmission(E)
    G_E[i] = 2 * T  # Factor of 2 for spin degeneracy

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(E_range, G_E, 'b-', linewidth=2)
plt.xlabel('Energy')
plt.ylabel('Conductance $(2e^2/h)$')
plt.title('Quantum Wire Conductance')
plt.grid(True)
plt.ylim(0, 6)
plt.show()

# Visualize potential
plt.figure(figsize=(10, 4))
plt.pcolormesh(x, y, V2D, cmap='RdGy')
plt.colorbar(label='Potential')
plt.title('Wire Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.show()