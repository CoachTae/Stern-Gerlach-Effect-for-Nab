import sys
import numpy as np
import Support

field = np.load('SG z-adjusted_m.npy')


test_coords = np.array([[0, 0, -0.133],
                       [0, 0, -0.139]])

nearest_idx = Support.find_nearest_points(test_coords, field)

print(field[nearest_idx])