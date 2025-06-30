import os
import re

import ctypes
import numpy as np

def read_constants_c(constants_c_path):
    """
    Read the constants.c file to extract some of the key constants used in RTGSFIT.
    """

    with open(constants_c_path, 'r') as file:
        content = file.read()

    # Extract N_R and N_Z
    n_r = int(re.search(r'const int N_R\s*=\s*(\d+);', content).group(1))
    n_z = int(re.search(r'const int N_Z\s*=\s*(\d+);', content).group(1))

    # Extract N_MEAS
    n_meas = int(re.search(r'const int N_MEAS\s*=\s*(\d+);', content).group(1))

    # Extact N_COIL
    n_coil = int(re.search(r'const int N_COIL\s*=\s*(\d+);', content).group(1))

    # Extact N_LCFS_MAX
    n_lcfs_max = int(re.search(r'const int N_LCFS_MAX\s*=\s*(\d+);', content).group(1))

    # Extract N_COEF
    n_coef = int(re.search(r'const int N_COEF\s*=\s*(\d+);', content).group(1))

    # Extract R_VEC array
    r_vec_match = re.search(r'const double R_VEC\[\]\s*=\s*{([^}]*)};', content, re.DOTALL)
    r_vec_str = r_vec_match.group(1)
    r_vec_list = list(map(float, r_vec_str.replace('\n', '').split(',')))
    r_vec = np.array(r_vec_list)

    # Extract Z_VEC array
    z_vec_match = re.search(r'const double Z_VEC\[\]\s*=\s*{([^}]*)};', content, re.DOTALL)
    z_vec_str = z_vec_match.group(1)
    z_vec_list = list(map(float, z_vec_str.replace('\n', '').split(',')))
    z_vec = np.array(z_vec_list)

    return n_r, n_z, n_meas, n_coil, n_lcfs_max, n_coef, r_vec, z_vec

def initial_flux_norm(r_vec, z_vec):
    """
    Create the initial array for the normalised flux.
    """

    r_grid, z_grid = np.meshgrid(r_vec, z_vec)
    opt_r = 1.5
    opt_z = 0.5
    flux_norm_radius = 0.25
    rho = np.sqrt((r_grid - opt_r)**2 + (z_grid - opt_z)**2)
    flux_norm = (1 - np.cos(np.pi * rho / flux_norm_radius)) / 2 * (rho <= flux_norm_radius) \
              + (rho > flux_norm_radius)
    
    return flux_norm
    
repo_path = os.path.dirname(os.path.abspath(__file__))
librtgsfit_path = '/home/alex.prokopyszyn/GitHub/rtgsfit/lib/librtgsfit.so'
# constants_c_path = '/home/alex.prokopyszyn/GitHub/rtgsfit/src/constants.c'
constants_c_path = os.path.join(repo_path, 'constants.c')
n_iters = 10

rtgsfit_lib = ctypes.CDLL(librtgsfit_path)

# Define the argument types for the rtgsfit function
rtgsfit_lib.rtgsfit.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # meas
    ctypes.POINTER(ctypes.c_double),  # coil_curr
    ctypes.POINTER(ctypes.c_double),  # flux_norm
    ctypes.POINTER(ctypes.c_int),     # mask
    ctypes.POINTER(ctypes.c_double),  # flux_total
    ctypes.POINTER(ctypes.c_double),  # error
    ctypes.POINTER(ctypes.c_double),  # lcfs_r
    ctypes.POINTER(ctypes.c_double),  # lcfs_z
    ctypes.POINTER(ctypes.c_int),     # lcfs_n
    ctypes.POINTER(ctypes.c_double),  # coef
    ctypes.POINTER(ctypes.c_double),  # flux_boundary
    ctypes.POINTER(ctypes.c_double)   # plasma_current
]
# Define the return type for the rtgsfit function
rtgsfit_lib.rtgsfit.restype = ctypes.c_int

n_r, n_z, n_meas, n_coil, n_lcfs_max, n_coef, r_vec, z_vec = read_constants_c(constants_c_path)
n_grid = n_r * n_z

meas = np.zeros(n_meas, dtype=np.float64)
coil_curr = np.zeros(n_coil, dtype=np.float64)
flux_norm = initial_flux_norm(r_vec, z_vec)
mask = np.ones(n_grid, dtype=np.int32)
flux_total = np.zeros(n_grid, dtype=np.float64)
coef = np.zeros(n_coef, dtype=np.float64)
error = np.array([0.1], dtype=np.float64) # PROKOPYSZYN: Change to 0.0
lcfs_r = np.zeros(n_lcfs_max, dtype=np.float64)
lcfs_z = np.zeros(n_lcfs_max, dtype=np.float64)
lcfs_n = np.array([0], dtype=np.int32)
flux_boundary = np.array([0.0], dtype=np.float64)
plasma_current = np.array([0.0], dtype=np.float64)

flux_total_output = np.zeros((n_iters, n_z, n_r))
flux_boundary_output = np.zeros((n_iters))
plasma_current_output = np.zeros((n_iters))

for i_iter in range(0, n_iters):

    meas = np.zeros(n_meas, dtype=np.float64)
    coil_curr = np.zeros(n_coil, dtype=np.float64)

    # Call the rtgsfit function
    result = rtgsfit_lib.rtgsfit(
        meas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coil_curr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flux_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        flux_total.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        error.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lcfs_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lcfs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lcfs_n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        coef.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flux_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        plasma_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    print("Iteration:", i_iter)
    print("Result:", result)
