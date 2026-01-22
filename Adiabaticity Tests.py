import sys
import numpy as np
import matplotlib.pyplot as plt
import Support
from Support import mn # import mass of neutron


#------------------------CONTROLS---------------------------------------------------------------
N = 10 # Number of neutrons
spin_orientation = 'random'
gravity = False
x0 = -1.19 # Starting x value for neutrons (m)
ymin, ymax = -0.03, 0.03 # Starting y value range for neutrons (m)
zmin, zmax = -0.035, 0.035 # Starting z value range for neutrons (m)
lambdamin, lambdamax = 2e-10, 25e-10# Wavelength range of neutrons being generated (m)
mu = Support.mu

# Set to None if we want this stuff to be generated randomly
rs = None
vs = None
wavelengths = None

# Custom parameters


#-----------------------------------------------------------------------------------------------

if gravity:
    g = 9.81
else:
    g = 0

#-------------------------------RANDOM NEUTRON VALUES--------------------------------
if rs is None:
    # Generate neutrons
    yz = np.random.uniform(low=[ymin, zmin], high=[ymax, zmax], size=(N,2)) # Starting y and z positions for neutrons
    rs = np.column_stack((np.full(N,x0), yz)) # Create position vectors from the same starting x0 and the randomly generated yzs
rs[:,2] -= 0.13189 # Offset for beam center being at 13.189cm below z=0

# Create a copy of starting positions to reference
r0s = rs.copy()

if wavelengths is None:
    # Give neutrons velocities based on wavelengths
    wavelengths = np.random.uniform(low=lambdamin, high=lambdamax, size=N)
if vs is None:
    xspeeds = Support.get_vs(wavelengths) # Convert wavelengths to velocities (m/s)
    # Turn 1D velocities array into Nx3 array (y and z velocities initialized at 0)
    vs = np.zeros((N,3))
    vs[:,0] = xspeeds

# Give the neutrons a random spin (up or down)
spins = Support.random_spin_directions(N) # Produces (N, 3) array of random spin vectors

#---------------------------LOAD FIELD------------------------------------------------
# Load field data (N, 7) where the 7 columns are [x, y, z, Bx, By, Bz, B]
#field_data = Support.custom_field([0,0,36.278], 0, 0, 1) # Custom field for testing
field_data = np.load('SG z-adjusted_m.npy')

counter = 0
angles = []
while True:
    # Increment counter
    counter += 1

    # Find the slice in x corresponding to each neutron
    nearest_idxs = Support.find_nearest_points(rs, field_data)

    # Find any indexing issues before next step. If there's an issue, a neutron is going out-of-bounds, so we must ignore it.
    within_x = (rs[:, 0] < 0.995) & (rs[:, 0] > -1.2) # +/- 120cm is our x limit
    within_y = abs(rs[:, 1]) < 0.05 # +/- 5cm is our y limit
    within_z = (abs(rs[:, 2]) < 0.18139) & (abs(rs[:,2]) > 0.08139) # +/- 5cm is our z limit from beam-center
    in_bounds = within_x & within_y & within_z

    # Exit condition: if all neutrons are out of bounds, stop the simulation
    if True not in in_bounds:
        break

    # Set velocities for any out-of-bounds neutrons to 0
    vs[~in_bounds] = 0


    # Calculate Force (F = grad(mu*B))
    F = np.zeros((N,3))
    dBxdx = (field_data[nearest_idxs[in_bounds] + 441, 3] - field_data[nearest_idxs[in_bounds] - 441, 3]) / 0.01 # Difference in Bx values. Shape should be (M,1) where M is len(in_bounds)
    dBydx = (field_data[nearest_idxs[in_bounds] + 441, 4] - field_data[nearest_idxs[in_bounds] - 441, 4]) / 0.01
    dBzdx = (field_data[nearest_idxs[in_bounds] + 441, 5] - field_data[nearest_idxs[in_bounds] - 441, 5]) / 0.01
    dBxdy = (field_data[nearest_idxs[in_bounds] + 21, 3] - field_data[nearest_idxs[in_bounds] - 21, 3]) / 0.01
    dBydy = (field_data[nearest_idxs[in_bounds] + 21, 4] - field_data[nearest_idxs[in_bounds] - 21, 4]) / 0.01
    dBzdy = (field_data[nearest_idxs[in_bounds] + 21, 5] - field_data[nearest_idxs[in_bounds] - 21, 5]) / 0.01
    dBxdz = (field_data[nearest_idxs[in_bounds] + 1, 3] - field_data[nearest_idxs[in_bounds] - 1, 3]) / 0.01
    dBydz = (field_data[nearest_idxs[in_bounds] + 1, 4] - field_data[nearest_idxs[in_bounds] - 1, 4]) / 0.01
    dBzdz = (field_data[nearest_idxs[in_bounds] + 1, 5] - field_data[nearest_idxs[in_bounds] - 1, 5]) / 0.01

    Fx = mu*spins[in_bounds, 0]*dBxdx + mu*spins[in_bounds, 1]*dBydx + mu*spins[in_bounds, 2]*dBzdx
    Fy = mu*spins[in_bounds, 0]*dBxdy + mu*spins[in_bounds, 1]*dBydy + mu*spins[in_bounds, 2]*dBzdy
    Fz = mu*spins[in_bounds, 0]*dBxdz + mu*spins[in_bounds, 1]*dBydz + mu*spins[in_bounds, 2]*dBzdz

    F[in_bounds] = np.stack([Fx, Fy, Fz], axis=1) # Should stack to an (M, 3) array where M = len(in_bounds)
    
    # Find the time it takes for the neutrons to get to the next slice in x (5mm in +x-hat direction)
    # Start with "no solution"
    t = np.full(vs[:,0].shape, np.inf, dtype=float)
    
    # Find the neutrons that have basically 0 force in the x-direction. We calculate their time differently to avoid division by 0
    no_Fx_neutrons = abs(F[:,0]) < 1e-36

    # Calculate no-x-force neutron times
    t[no_Fx_neutrons][in_bounds[no_Fx_neutrons]] = 0.005 / vs[no_Fx_neutrons,0][in_bounds[no_Fx_neutrons]] # First filter for neutrons with no x-force, then filter for in-bounds neutrons

    # Now calculate times for neutrons with a non-zero force
    t1 = mn * (-vs[:,0] + np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0] # We use [:,0] to only take the x-component of vectors and forces
    t2 = mn * (-vs[:,0] - np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0]

    # Set imaginary times to infinity
    imag_times1 = np.isnan(t1)
    imag_times2 = np.isnan(t2)
    t1[imag_times1] = np.inf
    t2[imag_times2] = np.inf

    # Set negative times to infinity so that we never pick them
    # First, make a mask of negative times
    neg_times1 = t1 < 0
    neg_times2 = t2 < 0

    # Set negative times to infinity
    t1[neg_times1] = np.inf
    t2[neg_times2] = np.inf

    # Choose the minimum positive time
    t = np.minimum(t1, t2)

    t[~no_Fx_neutrons] = np.minimum(t1[~no_Fx_neutrons], t2[~no_Fx_neutrons])

    # For time-tracking purposes, turn off time accumulation for out-of-bounds neutrons
    t[~in_bounds] = 0


    # Determine angle of gyration (modulo 2pi)
    B = np.zeros(N)
    B[in_bounds] = field_data[nearest_idxs[in_bounds], 6]
    num_gyr = Support.gammaHz * B * t # Number of full gyrations per neutron
    theta = (Support.gamma * B * t) % (2*np.pi) # Results should be the angle change for each neutron after gyrations

    # Find the unit vectors for which the neutrons gyrate about
    k = np.zeros(rs.shape) # Just create a (N, 3) array of zeros
    Bvec = np.zeros((N,3))
    Bvec[in_bounds] = field_data[nearest_idxs[in_bounds], 3:6]
    k[in_bounds] = Bvec[in_bounds] / B[in_bounds, None] # Normalize B vectors to get unit vectors

    # Update angle change at the end of the step with Rodrigues' rotation formula
    term1 = (spins.transpose()*np.cos(theta)).transpose()
    term2 = (Support.cross(k, spins).transpose()*np.sin(theta)).transpose()
    term3 = (k.transpose()*Support.dot(k, spins)*(1 - np.cos(theta))).transpose()
    spins = term1 + term2 + term3

    # Update positions
    rs += (vs * t[:, None]) + (0.5 * (F/mn) * t[:, None]**2)

    # Update velocities
    vs += (F/mn) * t[:, None]

    if counter % 20 == 0:
        #print(f"Neutrons have travelled: {counter*0.5}cm")
        pass
    
    dotprod = Support.dot(k, spins)
    angles.append(np.arccos(dotprod)) # No need to worry about magnitudes since k and spins should be unit vectors

angles = np.stack(angles, axis=1) # rows are neutrons, columns are their angles at various steps
anglediffs = np.copy(angles)
anglediffs[:, 1:] = (angles[:, 1:].transpose() - angles[:, 0]).transpose() # Create differences using initial angle as a reference
numsteps = anglediffs.shape[1]
print("Max Angle Differences (in degrees):")
print(np.max(anglediffs[:, :numsteps-2], axis=1)*(180/np.pi))


    
