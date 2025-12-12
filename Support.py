import scipy
import numpy as np

e = 1.60217663e-19
hbar = scipy.constants.hbar
h = hbar * 2 * np.pi
mp = 1.67262192e-27
mn = 1.6749275e-27
muN = (e * hbar) / (2 * mp)
mu = 1.9103 * muN

def random_spin_directions(n_spins):
    """
    Generate random spin vectors for spin-1/2 particles.
    Magnitude = h-bar/2, orientation uniformly distributed over the sphere.

    returns: Nx3 np array of spin vectors
    """

    # Random directions on the sphere
    u = np.random.rand(n_spins)
    v = np.random.rand(n_spins)

    theta = np.arccos(1 - 2 * u)  # polar angle
    phi = 2 * np.pi * v  # azimuthal angle

    # Convert spherical to cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    spins = np.stack((x, y, z), axis=-1)

    return spins


def get_vs(wavelengths):
    """
    Uses de Broglie wavelength equation to get neutron velocity.

    return: velocities in m/s
    """
    vs = h / (mn * wavelengths)
    return vs

def find_nearest_points(rs, field_data):
    start_idx = (int(200*rs[0,0]) + 240) * 21**2 # 21^2 points per slice, the parentheses is the offset such that -1.20 points at index 0, -1.195 to index 21^2, and so on
    end_idx = (int(200*rs[0,0]) + 241) * 21**2 # Original written equation was ((x + 1.2) / 0.005) * 21^2

    # Find the nearest point (index) for each neutron
    ys_neut = rs[:, 1][:, None] # Grab the y and z coordinates for each neutron
    zs_neut = rs[:, 2][:, None] # 2nd bracket reshapes from (N,) to (N, 1) to allow for broadcasting

    ys_grid = field_data[start_idx:end_idx, 1][None, :] # Grab y and z values from the field grid
    zs_grid = field_data[start_idx:end_idx, 2][None, :] # Again, 2nd bracket reshapes from (M,) to (1, M) for broadcasting

    y_diffs = ys_neut - ys_grid # Shape (N, M)
    z_diffs = zs_neut - zs_grid # Shape (N, M)

    distances = np.sqrt(y_diffs**2 + z_diffs**2) # distances[i,j] = distance between i'th neutron and grid point j in the start_idx:end_idx slice

    nearest_in_slice = np.argmin(distances, axis=-1) # Shape (N,)

    nearest_idxs = nearest_in_slice + start_idx # (N,) array of indices. Each value is the index in field_data that corresponds to the nearest point to the i'th neutron

    return nearest_idxs
#----------------------------OPERA FUNCTIONS----------------------------------------------------------------------
def table_to_npy(files):
    '''
    Files can be a single file or a list/tuple of files

    EXPECTS UNITS OF T, NOT GAUSS
    '''
    def write_array(filename):
        with open(filename, 'r') as file:
            # Skip lines until we finish the header
            while True:
                line = file.readline().strip()
                if line == '0':
                    break
            data = np.loadtxt(file)
            output_file = filename.replace('table', 'npy')
            np.save(output_file, data)

    if isinstance(files, str):
        write_array(files)

    elif isinstance(files, list) or isinstance(files, tuple):
        for file in files:
            write_array(file)
    else:
        print("Error in filename for table_to_npy.")
        print(f"Type given: {type(files)}")


def npy_to_table(array, filename):
    with open(filename, 'w') as file:
        file.write(str(len(array[:,0])) + ' 1 1 2\n')
        file.write('1 X [CM]\n')
        file.write('2 Y [CM]\n')
        file.write('3 Z [CM]\n')
        file.write('0\n')

        for point in array:
            file.write(f'{point[0]}\t{point[1]}\t{point[2]}\n')

if __name__ == "__main__":
    data = np.load('SG z-adjusted_m.npy')
    print(data[:3])