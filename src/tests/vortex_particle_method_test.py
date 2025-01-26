import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
# ==============================================================================
# Parameters
# ==============================================================================
U_inf = 1.0                # Freestream velocity (m/s)
rho = 1.225                # Air density (kg/m³)
nu = 1e-5                  # Kinematic viscosity (m²/s)
dt = 0.05                  # Time step (s)
total_time = 5.0           # Total simulation time (s)
chord_length = 1.0         # Chord length (m)
shed_interval = 0.1        # Vortex shedding interval (s)
vortex_core_radius = 0.01  # Avoid singularities (m)

# ==============================================================================
# Load Contour Geometry (replace with your .dat file)
# ==============================================================================
def load_contour(filename):
    data = np.loadtxt(filename, skiprows=1)
    x, y = data[:, 0], data[:, 1]
    return x, y

# Example: NACA 0012 airfoil (ensure trailing edge is the last point)
x, y = load_contour(os.getcwd() + "/UIUC_aerofoils/" + UDB['a18sm_dat'])

# ==============================================================================
# Preprocess Contour into Panels
# ==============================================================================
n_panels = len(x) - 1
panels = []
s = 0.0  # Arc length from leading edge
for i in range(n_panels):
    x1, y1 = x[i], y[i]
    x2, y2 = x[i+1], y[i+1]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    tx = (x2 - x1) / length
    ty = (y2 - y1) / length
    nx = -ty
    ny = tx
    panels.append({
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
        'tx': tx, 'ty': ty,
        'nx': nx, 'ny': ny,
        'length': length,
        's_start': s,  # Arc length from leading edge
        's_mid': s + length/2
    })
    s += length

# Identify trailing edge (last panel)
trailing_edge = {'x': x[-1], 'y': y[-1]}

# ==============================================================================
# Vortex Particle Simulation with Form Drag
# ==============================================================================
vortices = []  # List of vortices: [{'x', 'y', 'gamma'}]
time_steps = int(total_time / dt)
cl_history = []
cd_history = []

for step in range(time_steps):
    t = step * dt
    
    # Step 1: Shed vortex at trailing edge (Kutta condition)
    if step % int(shed_interval / dt) == 0:
        u_te, v_te = 0.0, 0.0
        for vortex in vortices:
            dx = trailing_edge['x'] - vortex['x']
            dy = trailing_edge['y'] - vortex['y']
            r_sq = dx**2 + dy**2 + vortex_core_radius**2
            u_te += (vortex['gamma'] * (-dy)) / (2 * np.pi * r_sq)
            v_te += (vortex['gamma'] * dx) / (2 * np.pi * r_sq)
        gamma_shed = -2 * np.pi * (u_te * trailing_edge['x'] + v_te * trailing_edge['y'])
        vortices.append({'x': trailing_edge['x'], 'y': trailing_edge['y'], 'gamma': gamma_shed})
    
    # Step 2: Advect vortices
    for vortex in vortices:
        u = U_inf
        v = 0.0
        for other in vortices:
            if vortex != other:
                dx = other['x'] - vortex['x']
                dy = other['y'] - vortex['y']
                r_sq = dx**2 + dy**2 + vortex_core_radius**2
                u += (other['gamma'] * (-dy)) / (2 * np.pi * r_sq)
                v += (other['gamma'] * dx) / (2 * np.pi * r_sq)
        vortex['x'] += u * dt
        vortex['y'] += v * dt
        vortex['gamma'] *= np.exp(-nu * dt / vortex_core_radius**2)  # Viscous decay
    
    # Step 3: Compute Lift (C_L) and Drag (C_D)
    total_circulation = sum([v['gamma'] for v in vortices])
    cl = (2 * total_circulation) / (U_inf * chord_length)
    cl_history.append(cl)
    
    # Compute skin friction drag (Cdf)
    Cdf = 0.0
    C_D_pressure = 0.0  # Form drag coefficient
    for panel in panels:
        # Compute velocity at panel midpoint (inviscid)
        x_mid = (panel['x1'] + panel['x2']) / 2
        y_mid = (panel['y1'] + panel['y2']) / 2
        u, v = U_inf, 0.0
        for vortex in vortices:
            dx = x_mid - vortex['x']
            dy = y_mid - vortex['y']
            r_sq = dx**2 + dy**2 + vortex_core_radius**2
            u_vortex = (-vortex['gamma'] * dy) / (2 * np.pi * r_sq)
            v_vortex = (vortex['gamma'] * dx) / (2 * np.pi * r_sq)
            u += u_vortex
            v += v_vortex
        
        # Tangential velocity component (edge velocity)
        U_e = u * panel['tx'] + v * panel['ty']
        
        # Skin friction (Schlichting correlation)
        Re_x = abs(U_e) * panel['s_mid'] / nu
        Cf = 0.0592 * Re_x ** (-0.2) if Re_x > 0 else 0.0
        Cdf += Cf * (panel['length'] / chord_length)
        
        # Pressure coefficient (Cp) and form drag
        Cp = 1 - (U_e / U_inf)**2
        C_D_pressure += Cp * panel['nx'] * (panel['length'] / chord_length)
    
    cd = Cdf + C_D_pressure
    cd_history.append(cd)
    
    # Step 4: Visualization (optional)
    if step % 10 == 0:
        plt.clf()
        plt.plot(x, y, 'k-', lw=2)
        plt.scatter([v['x'] for v in vortices], [v['y'] for v in vortices], c=[v['gamma'] for v in vortices], cmap='coolwarm', alpha=0.6)
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 0.5)
        plt.title(f"Time = {t:.2f}, $C_L$ = {cl:.2f}, $C_D$ = {cd:.3f}")
        plt.colorbar(label='Vortex Strength (Γ)')
        plt.pause(0.01)

# ==============================================================================
# Plot Results
# ==============================================================================
plt.figure()
plt.plot(np.arange(time_steps) * dt, cl_history, label='$C_L$')
plt.plot(np.arange(time_steps) * dt, cd_history, label='$C_D$')
plt.xlabel('Time')
plt.ylabel('Coefficient')
plt.legend()
plt.grid(True)
plt.show()