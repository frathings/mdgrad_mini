from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch
import matplotlib.animation as animation

import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import questionary
from tkinter import messagebox 

from abc import ABC, abstractmethod
from tkinter import messagebox 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import customtkinter as ctk
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch


class PDE(ABC):
    """     
    Abstract class for defining a generic PDE solver.
    """

    def __init__(self, L: float, N: int, dt: float, Tstep: int, dim: int = 2):
        assert dim in (1, 2, 3), "Dimension must be 1, 2, or 3."

        # Basic grid parameters
        self.dim = dim
        self.L = L
        self.N = N
        self.dt = dt
        self.Tstep = Tstep
        self.dx = L / N

        # Select computation device (CPU/GPU)
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

        # Initialize solution tensor T
        shape = (N,) if dim == 1 else (N, N) if dim == 2 else (N, N, N)
        self.T = torch.zeros(*shape, Tstep, device=self.device)

        # Create spatial grid
        if dim >= 2:
            self.x = np.linspace(0, L, N)
            self.y = np.linspace(0, L, N)
            self.X, self.Y = np.meshgrid(self.x, self.y)
        if dim == 3:
            self.z = np.linspace(0, L, N)

    @abstractmethod
    def evolve_timestep(self, t: int):
        """
        Abstract method to define how the PDE evolves in time.
        Subclasses must implement this method.
        """
        pass

    def solve(self):
        """Solve the PDE iteratively over all timesteps."""
        for t in range(1, self.Tstep):
            self.evolve_timestep(t)

    def set_initial_condition(self, func=None):
        """Sets initial conditions using a user-defined function."""
        if func:
            indices = torch.linspace(0, self.L, self.N, device=self.device)
            if self.dim == 1:
                self.T[:, 0] = func(indices)
            elif self.dim == 2:
                X, Y = torch.meshgrid(indices, indices, indexing='ij')
                self.T[..., 0] = func(X, Y)
            elif self.dim == 3:
                X, Y, Z = torch.meshgrid(indices, indices, indices, indexing='ij')
                self.T[..., 0] = func(X, Y, Z)

    def set_boundary(self, val=0):
        """Set boundary conditions (default: zero Dirichlet)."""
        if self.dim == 1:
            self.T[0, :] = val
            self.T[-1, :] = val
        elif self.dim == 2:
            self.T[0, :, :] = val
            self.T[-1, :, :] = val
            self.T[:, 0, :] = val
            self.T[:, -1, :] = val
        elif self.dim == 3:
            self.T[:, 0, :, :] = val
            self.T[:, -1, :, :] = val

class HeatEquation(PDE):
    """Solves the Heat Equation using an explicit finite difference method."""
    def __init__(self, L, N, D, dt, Tstep, dim, D_type='Constant'):
        super().__init__(L, N, dt, Tstep, dim)
        self.D = D
        if dim == 1:
            if D_type == "Linear":
                self.D_arr = torch.linspace(0, self.D, N, device=self.device)
            elif D_type == "Constant":
                self.D_arr = torch.full((N,), self.D, device=self.device)
        elif dim == 2:
            if D_type == "Linear":
                x = torch.linspace(0, self.D, N, device=self.device)
                self.D_arr = x.unsqueeze(1).repeat(1, N)  # Horizontal variation
            elif D_type == "Constant":
                self.D_arr = torch.full((N, N), self.D, device=self.device)

        else:  # 3D case
            self.D_arr = self.D  # Keep as a scalar (not implemented fully)

        self.a = self.dt * self.D_arr / (self.dx ** 2)  # Stability coefficient

    def evolve_timestep(self, t: int):
        """Explicit finite difference scheme for the heat equation."""
        if self.dim == 1:
            T_t = self.T[:, t - 1]  # Prendi il passo temporale precedente
            self.T[1:-1, t] = T_t[1:-1] + self.a[1:-1] * (T_t[:-2] - 2 * T_t[1:-1] + T_t[2:])
            self.T[1:-1, t] += (T_t[2:] - T_t[1:-1]) / self.dx * (self.a[2:] - self.a[1:-1])

        elif self.dim == 2: 
            T_t = self.T[..., t - 1]
            self.T[1:-1, 1:-1, t] = T_t[1:-1, 1:-1] + self.a[1:-1, 1:-1] * (
                            T_t[:-2, 1:-1] +
                            T_t[2:, 1:-1] +
                            T_t[1:-1, :-2] +
                            T_t[1:-1, 2:] -
                            4 * T_t[1:-1, 1:-1]) +\
                            (T_t[2:, 1:-1] - T_t[1:-1, 1:-1])/self.dx * (
                            self.a[2:, 1:-1] - self.a[1:-1, 1:-1]) +\
                            (T_t[1:-1, 2:] - T_t[1:-1, 1:-1])/self.dx * (
                            self.a[1:-1, 2:] - self.a[1:-1, 1:-1])  
        elif self.dim == 3:
            T_t = self.T[..., t - 1]
            self.T[1:-1, 1:-1, 1:-1, t] = T_t[1:-1, 1:-1, 1:-1] + self.a * (
                T_t[2:, 1:-1, 1:-1] + 
                T_t[:-2, 1:-1, 1:-1] +
                T_t[1:-1, 2:, 1:-1] + 
                T_t[1:-1, :-2, 1:-1] +
                T_t[1:-1, 1:-1, 2:] + 
                T_t[1:-1, 1:-1, :-2] -
                6 * T_t[1:-1, 1:-1, 1:-1]
            )


class PlotHandler:
    def __init__(self, solver):
        self.solver = solver  # it is HeatEquation class
    
    def plot_1d(self, time_steps=None):
        # predefinito step
        if time_steps is None:
            time_steps = [0, self.solver.Tstep // 2, self.solver.Tstep - 1]
                
        plt.figure(figsize=(10, 6))
        
        x = np.linspace(0, self.solver.L, self.solver.N)  # x coordinate array / serve per plottare la soluzione analitica
        
        # Plot 
        for t in time_steps:
            plt.plot(
                x,  
                self.solver.T[:, t].cpu().numpy(), # va convertito T da torch a numpy per il plotting
                label=f"t={t * self.solver.dt:.2f}s" 
            )
                
            # D costante -> soluzione analitica
            if torch.all(self.solver.D_arr == self.solver.D_arr[0]):
                D = float(self.solver.D_arr[0].cpu().numpy())  # Get D 
                t_val = t * self.solver.dt # get time
                analytical = np.sin(np.pi * x / self.solver.L) * np.exp(-D * np.pi**2 * t_val / self.solver.L**2) # soluzione analitica
                # plot tutto
                plt.plot(
                    x,
                    analytical,
                    '--',
                    label=f"Analytical t={t_val:.2f}s"
                )
                    
                #  maximum error
                numerical = self.solver.T[:, t].cpu().numpy()
                max_error = np.max(np.abs(numerical - analytical))
                print(f"Maximum error at t={t_val:.2f}s: {max_error:.6f}")

        plt.xlabel("Position")
        plt.ylabel("Temperature")
        plt.title("Temperature Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

    def animate_2d(self, interval=50):
        # Configura la figura e gli assi
        def _setup_figure():
            fig = plt.figure(figsize=(10, 8))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 10])
            controls_ax = fig.add_subplot(gs[0, :])
            main_ax = fig.add_subplot(gs[1, :])
            return fig, controls_ax, main_ax

        # PDE
        def _setup_main_ax(main_ax):
            im = main_ax.imshow(
                self.solver.T[..., 0].cpu(),
                cmap='plasma',
                extent=[0, self.solver.L, 0, self.solver.L],
                vmin=0,
                vmax=100
            )
            plt.colorbar(im, ax=main_ax, label='Temperature')
            return im

        # permette di aggiungere picchi al click
        # https://stackoverflow.com/questions/25521120/store-mouse-click-event-coordinates-with-matplotlib
        def _onclick(event):
            if event.inaxes == main_ax and state['paused']:
                y_idx = int(event.xdata / self.solver.L * (self.solver.N - 1))
                x_idx = int((self.solver.L - event.ydata) / self.solver.L * (self.solver.N - 1))
                x_idx = max(1, min(x_idx, self.solver.N-2))
                y_idx = max(1, min(y_idx, self.solver.N-2))
                _add_temperature_peak(x_idx, y_idx, state['current_frame'])

        # aggiunge  picco 
        def _add_temperature_peak(x_idx, y_idx, current_frame):
            peak_value = 100.0
            radius = 3
            for frame in range(current_frame, self.solver.Tstep): # applica a tutti i frame successivi 
            # griglia NxN -> i e j scorrono i punti della grid 
                for i in range(max(1, x_idx-radius), min(self.solver.N-1, x_idx+radius+1)):  # qui parte da sinistra evitando di uscire dalla grid o di colpire boundaruy
                    for j in range(max(1, y_idx-radius), min(self.solver.N-1, y_idx+radius+1)): # y 
                        dist = np.sqrt((i-x_idx)**2 + (j-y_idx)**2) # calcola la distanza tra le coordinate i,j attuali e quelle cliccate
                        if dist <= radius: 
                            temp = peak_value * np.exp(-dist**2/(radius/2)**2) # temp da aggiornare
                            if frame == current_frame:
                                self.solver.T[i, j, frame] = max( # se già è a temp alta non la cambia
                                    self.solver.T[i, j, frame], 
                                    torch.tensor(temp)
                                ) 
            for frame in range(current_frame + 1, self.solver.Tstep):  #  ricalcola considerato il picco
                self.solver.evolve_timestep(frame)   # è il for del solve
            im.set_array(self.solver.T[..., current_frame].cpu().numpy())
            plt.draw()

        # play/pausa
        def _on_play_pause(event):
            state['paused'] = not state['paused']
            if state['paused']:
                state['animation'].pause()
                play_button.label.set_text('Play')
            else:
                state['animation'].resume()
                play_button.label.set_text('Pause')

        # aggiorna i frame dell'animazione
        def _update(frame):
            if frame > state['current_frame']:
                self.solver.evolve_timestep(frame)
            im.set_array(self.solver.T[..., frame].cpu().numpy())
            main_ax.set_title(f"t={frame * self.solver.dt:.3f}s")
            state['current_frame'] = frame
            return [im]

        # salta a frame 
        def _set_frame(frame_percentage):
            frame = int(self.solver.Tstep * frame_percentage)
            if state['animation']:
                state['animation'].frame_seq = iter(range(frame, self.solver.Tstep))
                if not state['paused']:
                    state['animation'].event_source.start()

        # configurazione iniziale
        fig, controls_ax, main_ax = _setup_figure()
        im = _setup_main_ax(main_ax)

        state = {
            'paused': False,
            'animation': None,
            'current_frame': 0
        }

        controls_ax.text( # riquadro testo blu
            0.05, 0.5, 
            'Premi Pause e clicca qualche punto della grid per aggiungere picchi di temperatura:', 
            transform=controls_ax.transAxes,
            verticalalignment='center',
            fontfamily='sans-serif',
            fontsize=10,
            color='#2F5D8C',
            weight='medium',
            bbox=dict(
                facecolor='#F0F4F8',
                edgecolor='#D1E2F2',
                boxstyle='round,pad=0.5',
                alpha=0.7
            )
        )
        controls_ax.set_xticks([])
        controls_ax.set_yticks([])
        controls_ax.set_frame_on(False)

        play_button = plt.Button(plt.axes([0.4, 0.92, 0.1, 0.05]), 'Pause')
        frame_button = plt.Button(plt.axes([0.51, 0.92, 0.1, 0.05]), '50%')
        frame_button1 = plt.Button(plt.axes([0.61, 0.92, 0.1, 0.05]), '90%')

        play_button.on_clicked(_on_play_pause)
        frame_button.on_clicked(lambda x: _set_frame(0.5))
        frame_button1.on_clicked(lambda x: _set_frame(0.9))

        fig.canvas.mpl_connect('button_press_event', _onclick)

        state['animation'] = animation.FuncAnimation(
            fig, 
            _update,
            frames=range(self.solver.Tstep),
            interval=interval,
            blit=False
        )

        plt.show()

    def create_3d_plot(self, time_step: int = 0):
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Take middle slice for Z axis
        mid_z = self.solver.N // 2
        Z = self.solver.T[:, :, mid_z, time_step].cpu().numpy()
        
        surf = ax.plot_surface(self.solver.X, self.solver.Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha=0.7)
        
        # Update contours
        ax.contour(self.solver.X, self.solver.Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
        
        ax.set_xlim3d(0., 1.)
        ax.set_ylim3d(0., 1.)
        ax.set_zlim3d(0., 100.)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Temperature")
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig, ax, surf
    
    def create_isosurface_plot(self, time_step: int = 0, val=50):
        """Creates a 3D isosurface plot with smooth surfaces"""
        T = self.solver.T[..., time_step].cpu().numpy()
        # Clear previous plot
        self.ax.clear()
        
        # Plot boundary planes explicitly
        N = T.shape[0]
        x, y, z = np.meshgrid(np.linspace(0, self.solver.L, N),
                            np.linspace(0, self.solver.L, N),
                            np.linspace(0, self.solver.L, N))
        
        # Plot only front and back faces
        faces = [
            (x[0,:,:], y[0,:,:], z[0,:,:], T[0,:,:]),  # Front face
            (x[-1,:,:], y[-1,:,:], z[-1,:,:], T[-1,:,:])  # Back face
        ]
        
        for face_x, face_y, face_z, temp in faces:
            surf = self.ax.plot_surface(face_x, face_y, face_z, 
                                    facecolors=plt.cm.jet(plt.Normalize(0, val)(temp)),
                                    alpha=0.9,  # Aumentato l'alpha per le BC
                                    rstride=N-1,  # Rimuove completamente la griglia
                                    cstride=N-1,  # Rimuove completamente la griglia
                                    linewidth=0,
                                    antialiased=True,
                                    shade=True)  # Migliora l'aspetto 3D
        
        # Plot interior isosurfaces
        num_isosurfaces = 8 # T 3D che si restringe
        percentiles = np.linspace(5, 95, num_isosurfaces)
        isovalues = np.percentile(T[T > 0.1], percentiles)
        
        for level in isovalues:
            try:
                verts, faces, _, _ = measure.marching_cubes(T, level, step_size=1)  # Ridotto step_size per superfici più smooth
                if len(faces) > 0:
                    verts = verts * self.solver.L / (self.solver.N - 1)
                    mesh = Poly3DCollection([verts[faces[i]] for i in range(len(faces))])
                    color = plt.cm.jet(level/val)
                    alpha = 0.0 if level < 0.1 else 0.3
                    mesh.set_facecolor(color)
                    mesh.set_edgecolor('none')
                    mesh.set_alpha(alpha)
                    self.ax.add_collection3d(mesh)
            except Exception as e:
                continue
        
        # Configure plot
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Temperature Distribution (t={time_step * self.solver.dt:.3f}s)')
        
        # Set axis limits
        self.ax.set_xlim(0, self.solver.L)
        self.ax.set_ylim(0, self.solver.L)
        self.ax.set_zlim(0, self.solver.L)
        
        # Set aspect ratio and view
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.view_init(elev=20, azim=0)
        
        return self.ax.collections

    def animate_isosurface(self, interval: int = 100, step: int = 1,val=50):
        """Animate the 3D isosurface plot with stable animation"""
        print("Starting 3D isosurface animation...")
        
        try:
            # Create figure and axis only once
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Add colorbar
            norm = plt.Normalize(
                self.solver.T.min().cpu().numpy(),
                self.solver.T.max().cpu().numpy()
            )
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
            plt.colorbar(sm, label='Temperature')
            
            # Add space for controls
            self.fig.subplots_adjust(top=0.9)
            
            # Animation state
            self.state = {
                'paused': False,
                'current_frame': 0
            }
            
            def update_frame(frame):
                """Update the isosurface plot for each frame"""
                try:
                    collections = self.create_isosurface_plot(frame,val)
                    return collections
                except Exception as e:
                    print(f"Error updating frame {frame}: {e}")
                    return []
            
            def on_play_pause(event):
                """Toggle animation play/pause"""
                self.state['paused'] = not self.state['paused']
                if self.state['paused']:
                    self.anim.pause()
                    play_button.label.set_text('Play')
                else:
                    self.anim.resume()
                    play_button.label.set_text('Pause')
            
            def set_frame(frame_percentage):
                """Jump to specific frame"""
                frame = int(self.solver.Tstep * frame_percentage)
                if hasattr(self, 'anim'):
                    self.anim.frame_seq = iter(range(frame, self.solver.Tstep))
                    if not self.state['paused']:
                        self.anim.event_source.start()
            
            # Create control buttons
            play_button = plt.Button(plt.axes([0.4, 0.95, 0.1, 0.04]), 'Pause')
            frame_50_button = plt.Button(plt.axes([0.51, 0.95, 0.1, 0.04]), '50%')
            frame_90_button = plt.Button(plt.axes([0.62, 0.95, 0.1, 0.04]), '90%')
            
            # Connect button events
            play_button.on_clicked(on_play_pause)
            frame_50_button.on_clicked(lambda x: set_frame(0.5))
            frame_90_button.on_clicked(lambda x: set_frame(0.9))
            
            # Create animation
            self.anim = animation.FuncAnimation(
                self.fig,
                update_frame,
                frames=range(0, self.solver.Tstep, step),
                interval=interval,
                blit=False,
                cache_frame_data=False  # Disable frame caching to save memory
            )
            
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create isosurface visualization: {e}")
            
        finally:
            # Cleanup
            plt.close('all')
            
            

def get_float_input(prompt, default):
    """Helper function to ask for float input with validation."""
    return float(questionary.text(prompt, default=str(default)).ask())

def solve_heat_equation_3d(params):
    """
    Solves 3D heat equation with option for different visualization methods
    """
    try:
        L = float(params['L'])
        N = int(params['N'])
        D = float(params['D'])
        dt = float(params['dt'])
        Tstep = int(params['Tstep'])
        viz_type = params.get('visualization', 'surface') 
        dx = L / N
        a = D * dt / (dx ** 2)

        solver = HeatEquation(
            L=L, 
            N=N, 
            D=D, 
            dt=dt, 
            Tstep=Tstep, 
            dim=3,
        )

        # condizioni inizali
        center = L/2  
        width = L/10  

        def initial_condition(x, y, z):
            temp = 0 * torch.ones_like(x)
            
            mask = (torch.abs(x - center) < width) & \
                  (torch.abs(y - center) < width) & \
                  (torch.abs(z - center) < width)
            
            #  central region
            temp[mask] = 0
            
            #  boundary conditions
            #temp[0, :, :] = 50    # Lato frontale
            #temp[-1, :, :] = 50   # Lato posteriore
            temp[:, 0, :] = 10    # Lato sinistro
            temp[:, -1, :] = 10   # Lato destro

            return temp

        # initial conditions and boundary
        solver.set_initial_condition(initial_condition)
        solver.set_boundary(val=10)
        solver.solve()

        plotter = PlotHandler(solver)
        plotter.animate_isosurface(interval=50, step=1,val=10)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    return solver
def main():
    print("\n== Welcome to the Interactive Heat Equation Solver ==\n")

    # Select 1D, 2D, or 3D Heat Equation
    dim = questionary.select(
        "Select the heat equation dimension:",
        choices=["1D", "2D", "3D"]
    ).ask()

    print("\n🔧 Enter Parameters (Press Enter for defaults)\n")

    # Ask user for parameter values (with defaults)
    L = get_float_input("Enter domain length (L):", 1)
    N = int(get_float_input("Enter grid size (N):", 100))
    D = get_float_input("Enter diffusion coefficient (D):", 1e-2)
    dt = get_float_input("Enter time step (dt):", 1e-4)
    Tstep = int(get_float_input("Enter number of time steps (Tstep):", 10000))

    # Only ask for D_type in 1D and 2D (not for 3D)
    if dim in ["1D", "2D"]:
        D_type = questionary.select(
            "Select the diffusion type:",
            choices=["Linear", "Constant"]
        ).ask()
    else:
        D_type = "Constant"  # 3D always uses Constant

    if dim == "1D":
        print("\n✅ Running 1D Heat Equation Solver...\n")

        solver = HeatEquation(L=L, N=N, D=D, dt=dt, Tstep=Tstep, dim=1, D_type=D_type)
        solver.set_initial_condition(lambda x: torch.sin(np.pi * x / solver.L))

        print("Solving...")
        solver.solve()  
        print("Solution completed!")

        plotter = PlotHandler(solver)
        plotter.plot_1d()

    elif dim == "2D":
        print("\n✅ Running 2D Heat Equation Solver...\n")

        solver = HeatEquation(L=L, N=N, D=D, dt=dt, Tstep=Tstep, dim=2, D_type=D_type)
        solver.set_initial_condition(func=lambda x, y: torch.zeros(solver.N, solver.N))
        solver.set_boundary(val=100)

        print("Solving...")
        solver.solve()  
        print("Solution completed!")

        plotter = PlotHandler(solver)
        plotter.animate_2d(interval=50)

    else:  # 3D Case (D_type is Constant)
        print("\n✅ Running 3D Heat Equation Solver...\n")

        params = {
            'L': L, 'N': N, 'D': D, 'dt': dt, 'Tstep': Tstep,
        }

        solver = solve_heat_equation_3d(params)

    print("\n🎉 Simulation complete!\n")

if __name__ == "__main__":
    main()
