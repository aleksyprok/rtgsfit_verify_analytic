import os
import re

import numpy as np
import matplotlib.pyplot as plt

import poisson_matrix
import greens_matrices

def compute_limit_idx_and_weights(r, z, lim_r, lim_z, n_intrp, n_lim):
    """
    Computes the indices and weights for interpolation at specified limiter points.
    Parameters:
    - r: 1D array of radial grid points.
    - z: 1D array of vertical grid points.
    - lim_r: 1D array of radial limiter points.
    - lim_z: 1D array of vertical limiter points.
    - n_intrp: Number of interpolation points per limiter point.
    - n_lim: Number of limiter points.
    Returns:
    - limit_idx: 1D array of indices for the interpolation points.
    - limit_w: 1D array of weights for the interpolation points.
    """

    n_r = len(r)
    n_lim = len(lim_r)
    limit_idx = np.zeros(n_lim * n_intrp, dtype=int)
    limit_w = np.zeros(n_lim * n_intrp, dtype=float)

    for i, (lr, lz) in enumerate(zip(lim_r, lim_z)):
        if lr < r[0] or lr > r[-1] or lz < z[0] or lz > z[-1]:
            raise ValueError(f"Limiter point ({lr}, {lz}) is out of bounds of the grid.")
        r_idx = np.searchsorted(r, lr) - 1
        z_idx = np.searchsorted(z, lz) - 1

        limit_idx[n_intrp * i + 0] = n_r * z_idx + r_idx  # (r_idx, z_idx)
        limit_idx[n_intrp * i + 1] = n_r * z_idx + r_idx + 1  # (r_idx + 1, z_idx)
        limit_idx[n_intrp * i + 2] = n_r * (z_idx + 1) + r_idx  # (r_idx, z_idx + 1)
        limit_idx[n_intrp * i + 3] = n_r * (z_idx + 1) + r_idx + 1  # (r_idx + 1, z_idx + 1)

        r0, r1 = r[r_idx], r[r_idx + 1]
        z0, z1 = z[z_idx], z[z_idx + 1]
        dr = r1 - r0
        dz = z1 - z0
        limit_w[n_intrp * i + 0] = (r1 - lr) * (z1 - lz) / (dr * dz)  # (r_idx, z_idx)
        limit_w[n_intrp * i + 1] = (lr - r0) * (z1 - lz) / (dr * dz)  # (r_idx + 1, z_idx)
        limit_w[n_intrp * i + 2] = (r1 - lr) * (lz - z0) / (dr * dz)  # (r_idx, z_idx + 1)
        limit_w[n_intrp * i + 3] = (lr - r0) * (lz - z0) / (dr * dz)  # (r_idx + 1, z_idx + 1)

    return limit_idx, limit_w

def generate_data_dictionary():
    """
    Generates a dictionary called `data_dictionary` that contains
    the constant name and values that will be used by the 
    `const_to_file` function that will generate the contants.c file
    used by RTGSFIT.
    """

    # Define the constants
    frac = 0.99
    mu_0 = 4 * np.pi * 1e-7  # Permeability of free space in H/m
    n_lcfs_max = 500
    n_pls = 3
    n_xpt_max = 50


    # Define the grid related quantities
    r_min = 0.75
    r_max = 2.25
    r_axis = 1.5
    z_axis = 0.0
    z_min = -1.5
    z_max = 1.5
    n_r = 33
    n_z = 65
    n_grid = n_r * n_z
    r_vec = np.linspace(r_min, r_max, n_r)
    z_vec = np.linspace(z_min, z_max, n_z)
    r_grid, z_grid = np.meshgrid(r_vec, z_vec)
    r_flat = r_grid.flatten()
    z_flat = z_grid.flatten()
    d_r = r_vec[1] - r_vec[0]
    d_z = z_vec[1] - z_vec[0]
    r_ltrb = np.concatenate(
        (
            [r_vec[0]],  # (bottom, left)
            r_vec[0] * np.ones(n_z - 2),  # traverse (bottom, left) to (top, left) (excluding corners)
            [r_vec[0]],  # (top, left)
            r_vec[1:-1],  # traverse (top, left) to (top, right) (excluding corners)
            [r_vec[-1]],  # (top, right)
            r_vec[-1] * np.ones(n_z - 2),  # traverse (top, right) to (bottom, right) (excluding corners)
            [r_vec[-1]],  # (bottom, right)
            np.flip(r_vec[1:-1]),  # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    z_ltrb = np.concatenate(
        (
            [z_vec[0]],               # (bottom, left)
            z_vec[1:-1],              # traverse (bottom, left) to (top, left) (excluding corners)
            [z_vec[-1]],              # (top, left)
            z_vec[-1] * np.ones(n_r - 2), # traverse (top, left) to (top, right) (excluding corners)
            [z_vec[-1]],              # (top, right)
            np.flip(z_vec[1:-1]),     # traverse (top, right) to (bottom, right) (excluding corners)
            [z_vec[0]],               # (bottom, right)
            z_vec[0] * np.ones(n_r - 2),  # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    # Check that len(r_ltrb) == 2 * n_r + 2 * n_z - 4
    if len(r_ltrb) != 2 * n_r + 2 * n_z - 4:
        raise ValueError("r_ltrb must have length 2 * n_r + 2 * n_z - 4.")
    if len(z_ltrb) != 2 * n_r + 2 * n_z - 4:
        raise ValueError("z_ltrb must have length 2 * n_r + 2 * n_z - 4.")
    inv_r_ltrb_mu0 = 1 / (r_ltrb * mu_0)
    inv_r_mu0 = 1 / (r_flat * mu_0)

    # Calculate limiter quantities
    # Define the limiter values as a set of points along a rectangle with
    # corners at (r_lim_min, z_lim_min), (r_lim_min, z_lim_max),
    # (r_lim_max, z_lim_max), and (r_lim_max, z_lim_min)
    r_lim_min = 1
    r_lim_max = 2
    z_lim_min = -1
    z_lim_max = 1
    n_z_lim = 10
    n_r_lim = 5
    n_limit = 2 * n_r_lim + 2 * n_z_lim - 4
    r_lim_unique = np.linspace(r_lim_min, r_lim_max, n_r_lim)
    z_lim_unique = np.linspace(z_lim_min, z_lim_max, n_z_lim)
    r_lim = np.concatenate(
        (
            [r_lim_unique[0]],                   # (bottom, left)
            r_lim_unique[0] * np.ones(n_z_lim),  # traverse (bottom, left) to (top, left) (excluding corners)
            [r_lim_unique[0]],                   # (top, left)
            r_lim_unique[1:-1],                  # traverse (top, left) to (top, right) (excluding corners)
            [r_lim_unique[-1]],                  # (top, right)
            r_lim_unique[-1] * np.ones(n_z_lim), # traverse (top, right) to (bottom, right) (excluding corners)
            [r_lim_unique[-1]],                             # (bottom, right)
            np.flip(r_lim_unique[1:-1]),         # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    z_lim = np.concatenate(
        (
            [z_lim_unique[0]],                   # (bottom, left)
            z_lim_unique[1:-1],                  # traverse (bottom, left) to (top, left) (excluding corners)
            [z_lim_unique[-1]],                  # (top, left)
            z_lim_unique[-1] * np.ones(n_r_lim), # traverse (top, left) to (top, right) (excluding corners)
            [z_lim_unique[-1]],                  # (top, right)
            np.flip(z_lim_unique[1:-1]),         # traverse (top, right) to (bottom, right) (excluding corners)
            [z_lim_unique[0]],                   # (bottom, right)
            z_lim_unique[0] * np.ones(n_r_lim),  # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    n_intrp = 4
    limit_idx, limit_weight = compute_limit_idx_and_weights(
        r_vec, z_vec, r_lim, z_lim, n_intrp, n_limit
    )
    mask_lim_2d = (r_grid >= r_lim_min) & (r_grid <= r_lim_max) & \
                  (z_grid >= z_lim_min) & (z_grid <= z_lim_max)
    mask_lim = mask_lim_2d.flatten()
    
    # Calculate poisson matrix quantities
    lower_band, upper_band, perm_idx = poisson_matrix.compute_lup_bands(r_vec, z_vec)

    # Generate the flux loop locations
    n_fl_r = 4
    n_fl_z = 8
    n_flux_loops = 2 * n_fl_r + 2 * n_fl_z - 4
    r_fl_min = 0.9
    r_fl_max = 2.1
    z_fl_min = -1.2
    z_fl_max = 1.2
    r_fl_vec = np.linspace(r_fl_min, r_fl_max, n_fl_r)
    z_fl_vec = np.linspace(z_fl_min, z_fl_max, n_fl_z)
    r_fl = np.concatenate(
        (
            [r_fl_min],                     # (bottom, left)
            r_fl_min * np.ones(n_fl_z - 2), # traverse (bottom, left) to (top, left) (excluding corners)
            [r_fl_min],                     # (top, left)
            r_fl_vec[1:-1],                 # traverse (top, left) to (top, right) (excluding corners)
            [r_fl_max],                     # (top, right)
            r_fl_max * np.ones(n_fl_z - 2), # traverse (top, right) to (bottom, right) (excluding corners)
            [r_fl_max],                     # (bottom, right)
            np.flip(r_fl_vec[1:-1]),        # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    z_fl = np.concatenate(
        (
            [z_fl_min],                     # (bottom, left)
            z_fl_vec[1:-1],                 # traverse (bottom, left) to (top, left) (excluding corners)
            [z_fl_max],                     # (top, left)
            z_fl_max * np.ones(n_fl_r - 2), # traverse (top, left) to (top, right) (excluding corners)
            [z_fl_max],                     # (top, right)
            np.flip(z_fl_vec[1:-1]),        # traverse (top, right) to (bottom, right) (excluding corners)
            [z_fl_min],                     # (bottom, right)
            z_fl_min * np.ones(n_fl_r - 2), # traverse (bottom, right) to (bottom, left) (excluding corners)
        )
    )
    n_flux_loops = len(r_fl)

    # Generate BP probe locations
    n_bp_r = 5
    n_bp_z = 10
    n_np_probes = 2 * (n_bp_r - 2) + 2 * (n_bp_z - 2)  # Number of non-corner BP probes
    r_bp_min = 0.9
    r_bp_max = 2.1
    z_bp_min = -1.2
    z_bp_max = 1.2
    r_bp_vec = np.linspace(r_bp_min, r_bp_max, n_bp_r)
    z_bp_vec = np.linspace(z_bp_min, z_bp_max, n_bp_z)
    r_bp_vec = r_bp_vec[1:-1]  # Exclude corners for BP probe locations
    z_bp_vec = z_bp_vec[1:-1]  # Exclude corners for BP probe locations
    r_bp = np.concatenate(
        (
            r_bp_min * np.ones(n_bp_z - 2), # (bottom, left) to (top, left)
            r_bp_vec,                       # (top, left) to (top, right)
            r_bp_max * np.ones(n_bp_z - 2), # (top, right) to (bottom, right)
            np.flip(r_bp_vec),              # (bottom, right) to (bottom, left)       
        )
    )
    z_bp = np.concatenate(
        (
            z_bp_vec,                       # (bottom, left) to (top, left)
            z_bp_max * np.ones(n_bp_r - 2), # (top, left) to (top, right)
            np.flip(z_bp_vec),              # (top, right) to (bottom, right)
            z_bp_min * np.ones(n_bp_r - 2), # (bottom, right) to (bottom, left)
        )
    )
    angle_bp = np.concatenate(
        (
            0.5 * np.pi * np.ones(n_bp_z -2),   # Point up on the left side
            np.zeros(n_bp_r - 2),               # Point right on the top side
            -0.5 * np.pi * np.ones(n_bp_z - 2), # Point down on the right side
            np.pi * np.ones(n_bp_r - 2),                # Point left on the bottom side
        )
    )
    n_b_probes = len(r_bp)

    # Generate one rogovski coil that just goes around the plasma
    # And one that goes around vessel
    n_rogowski_coils = 2
    n_meas = n_flux_loops + n_b_probes + n_rogowski_coils

    # Generate passive/vessel coordinates
    # We suppose we have a vessel of infinite resistance except
    # at n_vessel points where we have wires of finite resistance
    n_vessel = 4
    r_vessel = np.array([0.95, 1.5, 2.05, 1.5])
    z_vessel = np.array([0, 1.1, 0, -1.1])
    n_coef = n_pls + n_vessel

    # Generate the weights for the measurements
    weights = np.ones(n_meas, dtype=np.float64)

    g_grid_meas_weight = greens_matrices.g_grid_meas_weight(
        r_vec, z_vec,
        r_fl, z_fl,
        r_bp, z_bp, angle_bp,
        n_rogowski_coils,
        n_meas,
        weights
    )

    g_coef_meas_weight = greens_matrices.g_coef_meas_weight(
        r_vessel, z_vessel,
        r_fl, z_fl,
        r_bp, z_bp, angle_bp,
        n_rogowski_coils,
        n_pls,
        n_coef,
        n_meas,
        weights)
    
    g_ltrb = greens_matrices.g_ltrb(r_ltrb, z_ltrb)

    # We have one PF coil but we set its current to zero
    n_pf_coils = 1
    g_meas_coil = np.zeros((n_meas, n_pf_coils), dtype=np.float64)
    g_grid_coil = np.zeros((n_grid, n_pf_coils), dtype=np.float64)

    g_grid_vessel = greens_matrices.g_grid_vessel(
        r_vessel, z_vessel,
        r_vec, z_vec)

    fig, ax = plt.subplots()
    ax.plot(r_ltrb, z_ltrb, label="Grid Boundary")
    ax.plot(r_lim, z_lim, label="Limiter Boundary")
    # ax.plot(r_pf_coil, z_pf_coil, label="PF Coil/Cylinder")
    ax.scatter(r_flat, z_flat, s=1, label="Grid Points", color='black')
    ax.scatter(r_lim, z_lim, s=2, label="Limiter Points", color='red')
    ax.scatter(r_flat[mask_lim], z_flat[mask_lim], s=1, label="MASK_LIM True Grid Points", color='blue')
    ax.scatter(r_bp, z_bp, s=2, label="BP Probe Points", color='green')
    ax.scatter(r_fl, z_fl, s=2, label="Flux Loop Points", color='orange')
    ax.scatter(r_vessel, z_vessel, s=2, label="Vessel Points", color='purple')
    # Add small lines for bp_probes with angles
    delta_bp_line = 0.05
    for r, z, angle in zip(r_bp, z_bp, angle_bp):
        dx = delta_bp_line * np.cos(angle)
        dy = delta_bp_line * np.sin(angle)
        ax.plot([r, r + dx], [z, z + dy], color='green', linewidth=0.5)
    ax.set_xlabel("Radial Position (r)")
    ax.set_ylabel("Vertical Position (z)")
    ax.set_title("Grid and Limiter Points")
    # Legend box outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(2.5, 1))
    ax.set_aspect('equal')
    # Plot in plots directory in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, "grid_and_limiter_points.png"),
                bbox_inches='tight', dpi=300)
    
    data_dictionary = {}
    data_dictionary["dr"] = d_r
    data_dictionary["dz"] = d_z
    data_dictionary["frac"] = frac 
    data_dictionary["inv_r_ltrb_mu0"] = inv_r_ltrb_mu0
    data_dictionary["inv_r_mu0"] = inv_r_mu0
    data_dictionary["limit_idx"] = limit_idx
    data_dictionary["limit_weight"] = limit_weight
    data_dictionary["lower_band"] = lower_band
    # convert mask lim from bool to int
    data_dictionary["mask_lim"] = mask_lim.astype(int)
    data_dictionary["n_bp_probes"] = n_b_probes
    data_dictionary["n_coef"] = n_coef
    data_dictionary["n_coil"] = n_pf_coils
    data_dictionary["n_flux_loops"] = n_flux_loops
    data_dictionary["n_grid"] = n_grid
    data_dictionary["n_intrp"] = n_intrp
    data_dictionary["n_lcfs_max"] = n_lcfs_max
    data_dictionary["n_limit"] = n_limit
    data_dictionary["n_ltrb"] = len(r_ltrb)
    data_dictionary["n_meas"] = n_meas
    data_dictionary["n_pls"] = n_pls
    data_dictionary["n_r"] = n_r
    data_dictionary["n_rogowski_coils"] = n_rogowski_coils
    data_dictionary["n_vess"] = n_vessel
    data_dictionary["n_xpt_max"] = n_xpt_max
    data_dictionary["n_z"] = n_z
    data_dictionary["perm_idx"] = perm_idx
    data_dictionary["r_grid"] = r_grid
    data_dictionary["r_mu0_dz2"] = r_flat * mu_0 * d_z**2
    data_dictionary["r_vec"] = r_vec
    data_dictionary["thresh"] = 1e-10 # numerical precision threshold
    data_dictionary["upper_band"] = upper_band
    data_dictionary["weight"] = weights
    data_dictionary["z_grid"] = z_grid
    data_dictionary["z_vec"] = z_vec
    data_dictionary["g_coef_meas_weight"] = g_coef_meas_weight
    data_dictionary["g_grid_coil"] = g_grid_coil
    data_dictionary["g_grid_meas_weight"] = g_grid_meas_weight
    data_dictionary["g_grid_vessel"] = g_grid_vessel
    data_dictionary["g_ltrb"] = g_ltrb
    data_dictionary["g_meas_coil"] = g_meas_coil

    return data_dictionary

def const_to_file(const_dict, file_read='constants.h', file_write='constants.c'):

    with open(file_read, 'r') as file:
        file_contents = file.read()

    # Define a function to perform variable replacements
    def replace_variables(match):

        assert((match.lastindex == 4) or (match.lastindex == 5))
        
        var_name = match.group(4).lower()
        var_name_in_file = " ".join([match.group(xx) for xx in range(2, 5)])
        if match.lastindex == 5:
            var_name_in_file = var_name_in_file + match.group(match.lastindex)
        
        if var_name in const_dict:
            replacement = const_dict[var_name]
            # Check the data type of the replacement value
            if isinstance(replacement, bool):
                return f'{var_name_in_file} = {int(replacement)};'
            elif isinstance(replacement, (int, float)):
                return f'{var_name_in_file} = {replacement};'
            elif isinstance(replacement, np.ndarray) and (match.lastindex == 5):
                data_list = map(str, replacement.flatten().tolist())
                data_list = [x if ((ii+1) % 20) else x+'\n' for ii, x in enumerate(data_list)]
                array_values = ', '.join(data_list)
                return f'{var_name_in_file} = {{{array_values}}};'
            else:
                raise Exception(f'{type(replacement)} is not catered for')
        else:
            raise Exception(f"{var_name} is not in dictionary")
            
        return match.group(0)

    pattern = r'\b(\w+)\s*\b(\w+)\s*\b(\w+)\s*\b(\w+)(\[\s*\d*\s*\])?\s*;'
    
    # Replace variables in the C code
    new_c_code = re.sub(pattern, replace_variables, file_contents)

    with open(file_write, 'w') as file:
        file.write(new_c_code)

if __name__ == "__main__":
    data_dictionary = generate_data_dictionary()
    const_to_file(data_dictionary, file_read='constants.h', file_write='constants.c')