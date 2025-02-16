import numpy as np
import scipy.sparse.linalg
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

def Potential(potential):
    # Define the potential V(x) according to a specific configuration given by 'potential'.
    
    V = np.zeros(Nx)
    
    if potential == 'infinite barrier':
        V[int(0.7*Nx):] = 1e10
    
    elif potential == 'infinite well':
        V[0:int(0.4*Nx)] = 1e10
        V[int(0.6*Nx)+1:] = 1e10
        
    elif potential == 'tunneling':
        V[int(0.7*Nx):int(0.8*Nx)] = 1e4
    
    elif potential == 'harmonic oscillator':
        V = 2e5*x**2
    
    else:
        print('Not a compatible potential, V=0')
    
    return V


def simulate(V):
    # Calculate the wave function psi according to the Crank-Nicolson method with Dirichlet boundary conditions.
    
    psi = np.zeros((Nt, Nx), dtype=complex)
    
    # Hamiltonian as a sparse matrix
    H = np.divide(scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray(), dx**2) + np.diag(V)
    I = scipy.sparse.identity(Nx).toarray()
    
    A = (I + np.multiply(H, 1j*dt/2)) # Left hand side matrix in eq. (4)
    
    # Initial Gaussian wave packet
    psi[0] = np.exp(-(x-x0)**2/(2*gaussWidth**2)) * np.exp(1j*x*waveVector)
    psi /= np.trapz(abs(psi[0])**2, dx=dx) # Normalisation
    
    for t in range(Nt-1):
        
        b = (I - np.multiply(H, 1j*dt/2)).dot(psi[t]) # Right hand side in eq. (4)
        
        # Use biconjugate gradient stabilized iteration to solve A*psi = b
        psi[t+1] = scipy.sparse.linalg.bicgstab(A, b.T)[0]
    
    return psi


def animate(t):
    linePsi.set_ydata(abs(psi[t])**2)
    return linePsi,


def init():
    linePsi.set_ydata(np.ma.array(x, mask=True))
    return linePsi,


def potential2D(potential):
    # Define the potential V(x,y) according to a specific configuration given by 'potential'.
    
    V2D = np.zeros((Nx, Nx))
    
    if potential == 'infinite barrier':
        V2D[:, int(0.7*Nx):] = 1e10
    
    elif potential == 'double slit':
        V2D[:int(0.3*Nx),            int(0.60*Nx):int(0.65*Nx)] = 1e10
        V2D[int(0.4*Nx):int(0.6*Nx), int(0.60*Nx):int(0.65*Nx)] = 1e10
        V2D[int(0.7*Nx):,            int(0.60*Nx):int(0.65*Nx)] = 1e10
    
    elif potential == 'tunneling':
        V2D[:, int(0.6*Nx):int(0.65*Nx)] = 1e3
    
    elif potential == 'harmonic oscillator':
        V2D = 2e5*(xx**2+yy**2)
    
    else:
        print('Not a compatible potential, V=0')
    
    return V2D


def simulation2D(V2D):
    # Calculate the wave function psi according to the Crank-Nicolson method with Dirichlet boundary conditions.
    
    psi2D = np.zeros((Nt, Nx, Nx), dtype=complex)
    
    # 1D Hamiltonian as a sparse matrix
    H = np.divide(scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray(), dx**2)
    
    I = scipy.sparse.identity(Nx).toarray()
    I2 = scipy.sparse.identity(Nx**2).toarray()
    
    H2D = np.kron(H, I) + np.kron(I, H) + np.diag(V2D.reshape((-1), order='F')) # 2D Hamiltonian
    
    A = (I2 + np.multiply(H2D, 1j*dt/2)) # Left hand side matrix in eq. (4)
    
    # Initial Gaussian wave packet
    psi2D[0] = np.exp(-(xx - x0)**2/(2*gaussWidthX**2) - (yy-  y0)**2/(2*gaussWidthY**2)) * np.exp(1j*(xx*waveVectorX 
                                                                                                       + yy*waveVectorY))
    psi2D[0] /= np.trapz(np.trapz(abs(psi2D[0])**2, dx=dx), dx=dx) # Normalisation
    
    psi2D = psi2D.reshape((Nt, -1), order='F')
    
    for t in range(Nt-1):
    
        b = (I2 - np.multiply(H2D, 1j*dt/2)).dot(psi2D[t]) # Right hand side in eq. (4)
        
        psi2D[t+1] = scipy.sparse.linalg.bicgstab(A, b.T)[0] # Use biconjugate gradient stabilized iteration to solve A*psi = b
    
    
    psi2D = psi2D.reshape((Nt, Nx, Nx), order='F')
    
    return psi2D


def animate2D(t):
    im = ax.contourf(xx, yy, abs(psi2D[t])**2, cmap=cm.gray)
    return im,
dx = 0.05 # Spacing between x values
dt = 0.001 # Spacing between time steps

x = np.arange(-1, 1, dx)
xx, yy = np.meshgrid(x, x)

Nx = len(x) # Number of spatial steps
Nt = 50 # Number of time steps


font = {'family' : 'serif',
        'size'   : 20}
plt.rc('font', **font)

# Choose one of the potentials: infinite barrier, double slit, tunneling, harmonic oscillator
V2D = potential2D('double slit')

# Parameters for the initial gaussian wave packet at t=0
gaussWidthX = 0.1
gaussWidthY = 0.8
waveVectorX = -20
waveVectorY = 0
x0 = -0.5 # Initial x-position
y0 = 0 # Initial y-position

psi2D = simulation2D(V2D)

fig = plt.figure(figsize=(10, 6))
im = plt.contourf(xx, yy, abs(psi2D[0])**2, cmap=cm.gray)
plt.contour(x, x, V2D, levels=[0], linewidths=3, colors='red')
ax = fig.gca()
plt.colorbar(im)

ani = animation.FuncAnimation(fig, animate2D, frames=Nt, interval=20, blit=True)
plt.show()