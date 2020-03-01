import numpy as np
import dill as pickle
import os
from datetime import datetime

def generate_triangles(n, x_scale, y_scale, x_shift_scale, y_shift_scale,
    datadir):
    # Randomly generates a dataset of right triangles which have been shifted
    # and rotated.
    #
    # Args:
    #  n: Number of triangles to generate.
    #  x_scale: Scale parameter for the x side of triangle.
    #  y_scale: Scale parameter for the y side of triangle.
    #  x_shift_scale: Scale parameter for the shift in the x-direction.
    #  y_shift_scale: Scale parameter for the shift in the y-direction.
    #  datadir: String indicating the directory where the data should be saved.
    # Returns:
    #  Nothing. Saves generated data as pickle file in the specified datadir.

    # Randomly generate data.
    x = np.stack([np.random.normal(scale=x_scale, size=n), np.zeros(n)])
    y = np.stack([np.zeros(n), np.random.normal(scale=y_scale, size=n)])
    z = np.zeros([2, n])
    rot = np.random.uniform(low=-np.pi, high=np.pi, size=n)
    x_shift = np.random.normal(scale=x_shift_scale, size=n)
    y_shift = np.random.normal(scale=y_shift_scale, size=n)
    shift = np.array([x_shift, y_shift])
    # Rotate points.
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
        [np.sin(rot), np.cos(rot)]])
    rot_mat = np.transpose(rot_mat, [2, 0, 1])
    for i in range(n):
        x[:,i] = np.matmul(rot_mat[i,:,:], x[:,i])
        y[:,i] = np.matmul(rot_mat[i,:,:], y[:,i])
    # Shift points.
    x = x + shift
    y = y + shift
    z = z + shift
    # Collect points.
    points = np.stack([x, y, z])
    points = np.transpose(points, (2, 0, 1))
    # Save data.
    time_str = datetime.now().strftime("%b_%m_%Y__%H_%M_%S")
    file_name = 'triangles_' + time_str + '.p'
    with open(os.path.join(datadir, file_name), 'wb') as f:
        pickle.dump({'points':points, 'n':n, 'x_scale':x_scale,
            'y_scale':y_scale, 'x_shift_scale':x_shift_scale,
            'y_shift_scale':y_shift_scale}, f)
