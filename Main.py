import sys
import numpy as np
import matplotlib.pyplot as plt
import Support
from Support import mn # import mass of neutron

dztotals = []
for dy in np.arange(-0.035, 0.036, 0.01):
    print(f"Current dy: {dy}")
    #------------------------CONTROLS---------------------------------------------------------------
    N = 7 # Number of neutrons
    spin_orientation = 'down'
    gravity = False
    x0 = -1.19 # Starting x value for neutrons (m)
    ymin, ymax = -0.03, 0.03 # Starting y value range for neutrons (m)
    zmin, zmax = -0.035, 0.035 # Starting z value range for neutrons (m)
    lambdamin, lambdamax = 2e-10, 25e-10# Wavelength range of neutrons being generated (m)
    mu = Support.mu
    dvspread = True # Set True if you only want the displacement across the decay volume
    xztracking = True

    # Set to None if we want this stuff to be generated randomly
    rs = []
    vs = None
    wavelengths = [5e-10]*N

    # Custom parameters
    z0s = np.array([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    for dz in z0s:
        r = [-1.19, dy, dz]
        rs.append(r)
    rs = np.array(rs)
    wavelengths=np.array(wavelengths)

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
    if spin_orientation == 'random':
        spins = np.random.choice([-1, 1], size=N)
    elif spin_orientation == 'up':
        spins = np.ones(N)
    elif spin_orientation == "down":
        spins = -np.ones(N)


    #---------------------------MANUAL NEUTRON VALUES-------------------------------------


    #---------------------------LOAD FIELD------------------------------------------------
    # Load field data (N, 7) where the 7 columns are [x, y, z, Bx, By, Bz, B]
    #field_data = Support.custom_field([0,0,36.278], 0, 0, 1) # Custom field for testing
    field_data = np.load('SG z-adjusted_m.npy')

    counter = 0
    if N > 1:
        max_dys = np.zeros(N) # Used to track maximum displacement in y
        max_dzs = np.zeros(N) # Used to track maximum displacement in z
        max_r_perp = np.zeros(N) # sqrt(dy^2 + dz^2)
    if xztracking:
        zs = [] # Stores all z-values for neutron tracking
        zs.append((rs[:,2] - r0s[:,2]).copy())
        xs = []
        xs.append(rs[:,0].copy())
        ttotals = np.zeros(N)
    while True:
        # Increment counter
        counter += 1

        # Find the slice in x corresponding to each neutron
        nearest_idxs = Support.find_nearest_points(rs, field_data)

        # Find any indexing issues before next step. If there's an issue, a neutron is going out-of-bounds, so we must ignore it.
        within_x = (rs[:, 0] < 1) & (rs[:, 0] > -1.2) # +/- 120cm is our x limit
        within_y = abs(rs[:, 1]) < 0.05 # +/- 5cm is our y limit
        within_z = (abs(rs[:, 2]) < 0.18139) & (abs(rs[:,2]) > 0.08139) # +/- 5cm is our z limit from beam-center
        in_bounds = within_x & within_y & within_z

        # Exit condition: if all neutrons are out of bounds, stop the simulation
        if True not in in_bounds:
            break

        # Set velocities for any out-of-bounds neutrons to 0
        vs[~in_bounds] = 0

        # Calculate the gradient of |B|
        dBdx = (field_data[nearest_idxs[in_bounds] + 441, 4] - field_data[nearest_idxs[in_bounds] - 441, 4]) / 0.01
        dBdy = (field_data[nearest_idxs[in_bounds] + 21, 5] - field_data[nearest_idxs[in_bounds] - 21, 5]) / 0.01
        dBdz = (field_data[nearest_idxs[in_bounds] + 1, 6] - field_data[nearest_idxs[in_bounds] - 1, 6]) / 0.01
        gradB = np.stack([dBdx, dBdy, dBdz], axis=1)

        
        # Calculate the force on each neutron
        F = np.zeros((N,3)) # To avoid shape issues on line 93 (in_bounds is shape (N,))
        Ftrans = gradB.transpose()*spins[in_bounds] # We transpose gradB for broadcasting purposes. We flip it back next line
        F[in_bounds] = mu*Ftrans.transpose() # Grad(B) is of shape (len(in_bounds),), not N. So we fix shape of spins to match
        F[:,2] += mn*g
        F[~in_bounds] = 0
    
        # Find the time it takes for the neutrons to get to the next slice in x (5mm in +x-hat direction)
        # Start with "no solution"
        t = np.full(vs[:,0].shape, np.inf, dtype=float)
        
        # Find the neutrons that have basically 0 force in the x-direction. We calculate their time differently to avoid division by 0
        no_Fx_neutrons = F[:,0] < 1e-36

        # Calculate no-x-force neutron times
        t[no_Fx_neutrons] = 0.005 / vs[no_Fx_neutrons,0]

        # Now calculate times for neutrons with a non-zero force

        t1 = mn * (-vs[:,0] + np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0] # We use [:,0] to only take the x-component of vectors and forces
        t2 = mn * (-vs[:,0] - np.sqrt(vs[:,0]**2 + (2*F[:,0]*0.005/mn))) / F[:,0]

        t[~no_Fx_neutrons] = np.maximum(t1[~no_Fx_neutrons], t2[~no_Fx_neutrons])

        # For time-tracking purposes, turn off time accumulation for out-of-bounds neutrons
        t[~in_bounds] = 0

        # Update total flight time
        if xztracking:
            ttotals += t

        # Update positions
        rs += (vs * t[:, None]) + (0.5 * (F/mn) * t[:, None]**2)

        # Update velocities
        vs += (F/mn) * t[:, None]

        if xztracking: # If we are tracking neutron paths
            xs.append(rs[:,0].copy())
            zs.append((rs[:,2] - r0s[:,2]).copy())

        if counter % 20 == 0:
            #print(f"Neutrons have travelled: {counter*0.5}cm")
            pass

        if counter % 1 == 0:
            # Check the spread from the starting positions
            if not dvspread:
                diffs = rs - r0s
            elif dvspread:
                if 0.1 - 1e-4 < rs[0,0] < 0.1 + 1e-4:
                    diffs = rs - r0s
                else:
                    continue
            dy = diffs[:,1]
            dz = diffs[:,2]
            r_perp = np.hypot(dy, dz)

            # Only update for alive neutrons
            max_dys[in_bounds] = np.maximum(max_dys[in_bounds], np.abs(dy[in_bounds]))
            max_dzs[in_bounds] = np.maximum(max_dzs[in_bounds], np.abs(dz[in_bounds]))
            max_r_perp[in_bounds] = np.maximum(max_r_perp[in_bounds], r_perp[in_bounds])

        
    # Individual neutron tracking
    '''print(f"Final velocity: {vs}")
    xs = np.array(xs)
    zs = np.array(zs)
    xs *= 100 # convert to cm
    zs *= 100 # convert to cm
    plt.plot(xs, zs)
    plt.xlabel('x (cm)')
    plt.ylabel('z (cm)')
    plt.show()
    sys.exit()'''

    # Histogram the max perpendicular displacements
    '''print(diffs)
    plt.figure()
    plt.hist(max_r_perp*1e6, bins=50, edgecolor='black')
    plt.xlabel('Maximum Perpendicular Displacement (um)', fontsize=18)
    plt.ylabel('Number of Neutrons', fontsize=18)
    plt.show()'''

    '''# Spread vs z0 plotting
    zs = np.array(zs)
    xs = np.array(xs)
    mask = (xs[:,0] > -0.12) & (xs[:,0] < 0.1)
    plt.figure()
    for i in range(7):
        plt.plot(xs[mask, i]*100, zs[mask, i]*1e6, label=f"{int(z0s[i]*100)}cm from beam center")
    plt.xlim(-10, 10)
    plt.title(f"{r0s[:,1][0]*100}cm lateral displacement from beam center, λ={wavelengths[0]*1e10:.1f}Å neutrons", fontsize=18)
    plt.xlabel("x (cm)", fontsize=18)
    plt.ylabel("dz from starting position (um)", fontsize=18)
    plt.legend()
    plt.show()'''

    # Plot max dz vs dy
    xs = np.array(xs)
    zs = np.array(zs)
    mask = (xs[:,0] > -0.12) & (xs[:,0] < 0.1)
    xs = xs[mask]
    zs = zs[mask]
    dztotal = zs[-1, :] - zs[0, :]
    dztotals.append(dztotal)
dztotals = np.array(dztotals)
print(dztotals.shape)