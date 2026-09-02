import sys
import numpy as np
import matplotlib.pyplot as plt
import Support
from Support import mn # import mass of neutron

DV_offset = 0.13189 # Offset for beam center being at 13.189cm below z=0

# Collimator dimensions (m)
C2x = -1.029 # Collimator 2 x-position
C2h = 0.07 # Collimator 2 height
C2w = 0.064 # Collimator 2 width

E1x = -0.558 # Entrance collimator x-position
E1h = 0.07 # Entrance collimator height
E1w = 0.064 # Entrance collimator width

E2x = -0.372 # Exit collimator x-position
E2h = 0.07 # Exit collimator height
E2w = 0.054 # Exit collimator width

E3x = -0.202 # Exit collimator x-position
E3h = 0.07 # Exit collimator height
E3w = 0.054 # Exit collimator width
#------------------------CONTROLS---------------------------------------------------------------
N = 10000 # Number of neutrons
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
rs[:,2] -= DV_offset # Offset for beam center being at 13.189cm below z=0

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

init_spins = spins.copy() # Create a copy of the initial spins to reference


#---------------------------LOAD FIELD------------------------------------------------
# Load field data (N, 7) where the 7 columns are [x, y, z, Bx, By, Bz, B]
#field_data = Support.custom_field([0,0,36.278], 0, 0, 1) # Custom field for testing
field_data = np.load('SG z-adjusted_m.npy')

counter = 0
neutrons_lost = 0
angles = []
while True:
    # Increment counter
    counter += 1

    # Find the slice in x corresponding to each neutron
    nearest_idxs = Support.find_nearest_points(rs, field_data)

    # Check of neutrons are hitting the collimators. If they are, set them out-of-bounds.
    median_x = np.median(rs[:,0])

    # Check for collimator 2
    y_check = np.abs(rs[:,1]) - C2w/2
    z_check = np.abs(rs[:,2] + DV_offset) - C2h/2
    mask = ((np.abs(median_x - C2x) < 5e-3) & 
            (((y_check >= 0) & (y_check < 20)) | 
             ((z_check > 0) & (z_check < 20))))
    rs[mask] = -100 # Sets positions well out of bounds so they can be caught by future in-bounds checks
    neutrons_lost += np.sum(mask)

    # Check for entrance collimator 1
    y_check = np.abs(rs[:,1]) - E1w/2
    z_check = np.abs(rs[:,2] + DV_offset) - E1h/2
    mask = ((np.abs(median_x - E1x) < 5e-3) & 
            (((y_check >= 0) & (y_check < 20)) | 
            ((z_check > 0) & (z_check < 20))))
    rs[mask] = -100
    neutrons_lost += np.sum(mask)

    # Check for entrance collimator 2
    y_check = np.abs(rs[:,1]) - E2w/2
    z_check = np.abs(rs[:,2] + DV_offset) - E2h/2
    mask = ((np.abs(median_x - E2x) < 5e-3) & 
            (((y_check >= 0) & (y_check < 20)) | 
            ((z_check > 0) & (z_check < 20))))
    rs[mask] = -100 
    neutrons_lost += np.sum(mask)

    # Check for entrance collimator 3
    y_check = np.abs(rs[:,1]) - E3w/2
    z_check = np.abs(rs[:,2] + DV_offset) - E3h/2
    mask = ((np.abs(median_x - E3x) < 5e-3) & 
            (((y_check >= 0) & (y_check < 20)) | 
            ((z_check > 0) & (z_check < 20))))
    rs[mask] = -100 
    neutrons_lost += np.sum(mask)

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
    t = np.full(N, np.inf)
    
    # Masks for neutrons without an Fx and one for one with an Fx (since we divide by Fx in the time calculation)
    no_Fx = np.abs(F[:, 0]) < 1e-36
    has_Fx = ~no_Fx

    # Essentially zero Fx
    mask = no_Fx & in_bounds
    t[mask] = 0.005 / vs[mask, 0]

    # Nonzero Fx
    mask = has_Fx & in_bounds

    # Discriminant calculation
    disc = vs[mask, 0]**2 + 2 * F[mask, 0] * 0.005 / mn 

    # Check that the discriminant is not imaginary
    valid = disc >= 0 

    # Create arrays of the size of the mask
    t1 = np.full(np.sum(mask), np.inf)
    t2 = np.full(np.sum(mask), np.inf)

    # Fill in those arrays with the possible solutions
    t1[valid] = mn * (
        -vs[mask, 0][valid] + np.sqrt(disc[valid])
    ) / F[mask, 0][valid]

    t2[valid] = mn * (
        -vs[mask, 0][valid] - np.sqrt(disc[valid])
    ) / F[mask, 0][valid]

    # Set negative times to infinity so that they are never chosen as the minimum
    t1[t1 < 0] = np.inf
    t2[t2 < 0] = np.inf

    t[mask] = np.minimum(t1, t2)

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


    
