import json
import os

"""
Created by:
Alexander Heilmeier

Created on:
12.10.2018

Documentation:
This script creates a vehicle parameter file. Please run it after changing any values!
"""

# set output file name
filename = "racecar.json"

# ----------------------------------------------------------------------------------------------------------------------
# GGV FILES ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# set name of ggv diagram file to use
ggv = "ggv.txt"

# ----------------------------------------------------------------------------------------------------------------------
# OPTIMIZATION OPTIONS -------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# stepsizes
stepsizes = {"stepsize_prep": 1.0,                  # [m] used for linear interpolation before spline approximation
             "stepsize_reg": 3.0,                   # [m] used for spline interpolation after spline approximation
             "stepsize_interp_after_opt": 2.0}      # [m] used for spline interpolation after optimization

# spline regression smooth options
reg_smooth_opts = {"k_reg": 3,                      # [-] order of BSplines -> standard: 3
                   "s_reg": 10}                     # [-] smoothing factor

# optimization problem options
optim_opts = {"w_veh": 3.4,                         # [m] used to calculate the allowed deviation from the centerline
              "tire_model_exp": 1.0,                # [-] exponent used in the acceleration model
              "window_size_conv_filt": 3,           # [-] set None if not used
              "kappa_bound": 0.1,                   # [rad/m] curvature boundary for optimization
              "iqp_iters_min": 3,                   # [-] minimum number of iterations for the IQP
              "iqp_curv_error_allowed": 0.01}       # [rad/m] maximum allowed curvature error for the IQP

# real vehicle dimensions for edge checks
veh_dims = {"l_veh_real": 4.7,                      # [m]
            "w_veh_real": 2.0}                      # [m]

# ----------------------------------------------------------------------------------------------------------------------
# SAVE PARAMETERS ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# set everything together
pars = {"ggv": ggv,
        "stepsizes": stepsizes,
        "reg_smooth_opts": reg_smooth_opts,
        "optim_opts": optim_opts,
        "veh_dims": veh_dims}

# get current path
module_path = os.path.dirname(os.path.abspath(__file__))

# write data to .json file
with open(module_path + "/" + filename, "w") as fh:
    json.dump(pars, fh, indent=4)
