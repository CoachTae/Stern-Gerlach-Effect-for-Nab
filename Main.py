import sys
import numpy as np
import matplotlib.pyplot as plt
import Support
from Support import mn # import mass of neutron


#------------------------CONTROLS---------------------------------------------------------------
N = 100000 # Number of neutrons
spin_orientation = 'random'
gravity = True
x0 = -1.19 # Starting x value for neutrons (m)
ymin, ymax = -0.03, 0.03 # Starting y value range for neutrons (m)
zmin, zmax = -0.035, 0.035 # Starting z value range for neutrons (m)
lambdamin, lambdamax = 2e-10, 25e-10# Wavelength range of neutrons being generated (m)
mu = Support.mu
#-----------------------------------------------------------------------------------------------

if gravity:
    g = 9.81
else:
    g = 0

# Generate neutrons
yz = np.random.uniform(low=[ymin, zmin], high=[ymax, zmax], size=(N,2)) # Starting y and z positions for neutrons
rs = np.column_stack((np.full(N,x0), yz)) # Create position vectors from the same starting x0 and the randomly generated yzs
rs[:,2] -= 0.13189 # Offset for beam center being at 13.189cm below z=0

# Create a copy of starting positions to reference
r0s = rs.copy()

# Give neutrons velocities based on wavelengths
wavelengths = np.random.uniform(low=lambdamin, high=lambdamax, size=N)
xspeeds = Support.get_vs(wavelengths) # Convert wavelengths to velocities (m/s)
# Turn 1D velocities array into Nx3 array (y and z velocities initialized at 0)
vs = np.zeros((N,3))
vs[:,0] = xspeeds

# Give the neutrons a random spin (up or down)
if spin_orientation == 'random':
    spins = np.random.choice([-1, 1], size=N)
elif spin_orientation == 'up':
    spins = np.ones(N)
elif spin_orientation == "down":
    spins = -np.ones(N)

# Load field data (N, 7) where the 7 columns are [x, y, z, Bx, By, Bz, B]
field_data = np.load('Stern-Gerlach 5mm spacing.npy')

counter = 0
max_dys = np.zeros(N)
max_dzs = np.zeros(N)
max_r_perp = np.zeros(N) # sqrt(dy^2 + dz^2)
while True:
    # Increment counter
    counter += 1

    # Find the slice in x corresponding to each neutron
    nearest_idxs = Support.find_nearest_points(rs, field_data)

    # Find any indexing issues before next step. If there's an issue, a neutron is going out-of-bounds, so we must ignore it.
    within_x = abs(rs[nearest_idxs, 0]) < 1.2 # +/- 120cm is our x limit
    within_y = abs(rs[nearest_idxs, 1]) < 0.05 # +/- 5cm is our y limit
    within_z = abs(rs[nearest_idxs, 2]) < 0.05 # +/- 5cm is our z limit
    in_bounds = within_x & within_y & within_z

    print(True in within_x)
    print(True in within_y)
    print(True in within_z)
    print(True in in_bounds)
    sys.exit()
    # Set velocities for any out-of-bounds neutrons to 0
    vs[~in_bounds] = 0

    # Calculate the gradient of |B|
    dBdx = (field_data[nearest_idxs[in_bounds] + 441, 6] - field_data[nearest_idxs[in_bounds] - 441, 6]) / 0.01
    dBdy = (field_data[nearest_idxs[in_bounds] + 21, 6] - field_data[nearest_idxs[in_bounds] - 21, 6]) / 0.01
    dBdz = (field_data[nearest_idxs[in_bounds] + 1, 6] - field_data[nearest_idxs[in_bounds] - 1, 6]) / 0.01
    gradB = np.stack([dBdx, dBdy, dBdz], axis=1)

    # Calculate the force on each neutron
    F = mu*gradB*spins[in_bounds] # Grad(B) is of shape (len(in_bounds),), not N. So fix shape of spins to match
    F[:,2] += mn*g
    F[~in_bounds] = 0

    # Find the time it takes for the neutrons to get to the next slice in x (5mm in +x-hat direction)
    try:
        t1 = mn * (-vs[:,0] + np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0] # We use [:,0] to only take the x-component of vectors and forces
    except:
        t1 = -1000 # If the above is sqrt of a negative number, just assign t1 to something obviously incorrect
        
    try:
        t2 = mn * (-vs[:,0] - np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0]
    except:
        t2 = -1000 # If sqrt is negative, assign to obviously incorrect

    t = np.maximum(t1, t2)

    if t == -1000:
        print("Error in time calculation. Exiting loop.")
        sys.exit()

    # Update positions
    rs += (vs * t[:, None]) + (0.5 * (F/mn) * t[:, None]**2)

    # Update velocities
    vs += (F/mn) * t[:, None]

    if counter % 1 == 0:
        # Check the spread from the starting positions
        diffs = rs - r0s
        dy = diffs[:,1]
        dz = diffs[:,2]
        r_perp = np.hypot(dy, dz)

        # Only update for alive neutrons
        max_dys[in_bounds] = np.maximum(max_dys[in_bounds], np.abs(dy[in_bounds]))
        max_dzs[in_bounds] = np.maximum(max_dzs[in_bounds], np.abs(dz[in_bounds]))
        max_r_perp[in_bounds] = np.maximum(max_r_perp[in_bounds], r_perp[in_bounds])

    # Exit condition: if all neutrons are out of bounds, stop the simulation
    if True not in in_bounds:
        break

# Histogram the max perpendicular displacements
plt.figure()
plt.hist(max_r_perp*1e3, bins=50, edgecolor='black')
plt.xlabel('Maximum Perpendicular Displacement (mm)')
plt.ylabel('Number of Neutrons')
plt.show()