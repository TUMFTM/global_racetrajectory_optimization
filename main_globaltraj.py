import opt_geometrical
import process_functions
import numpy as np
import time
import json
import os
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
import configparser
import pkg_resources

"""
Created by:
Alexander Heilmeier

Created on:
31.01.2019

Documentation:
This script has to be executed to generate an optimal trajectory based on a given reference track.
"""

# ----------------------------------------------------------------------------------------------------------------------
# USER INPUT -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

file_paths = {}

# choose vehicle parameter file
file_paths["veh_params_file"] = "racecar.ini"

# debug and plot options
debug = True                                        # console messages
plot_opts = {"opt_min_curv": False,                 # plot curvature based on original linearization and solution based
             "raceline": True,                      # plot optimized path
             "curv_profile": True,                  # plot curvature profile
             "velprofile": True,                    # plot velocity profile
             "velprofile_3d": False,                # plot 3D velocity profile above raceline
             "velprofile_3d_stepsize": 1.0,         # [m] vertical lines stepsize in 3D velocity profile plot
             "spline_normals": False,               # plot spline normals
             "mintime": False}                      # plot states, controls, tire forces if opt_mintime = True

# select track file (including centerline coords + track widths)
file_paths["track_file"] = "Berlin_2018.csv"
# file_paths["track_file"] = "HandlingTrack.csv"
# file_paths["track_file"] = "roundedRectangle.csv"

# set import options
# Berlin: set_new_start 106.0, 141.0
imp_opts = {"flip_imp_track": False,                    # flip imported track to reverse direction
            "set_new_start": True,                      # set new starting point (changes order, not coordinates)
            "new_start": np.array([106.0, 141.0])}      # [x_m, y_m]

# check normal vector crossings (can take a while)
check_normal_crossings = False

# optimization use switches (minimum curvature/shortest path)
use_opt_mincurv = False             # minimum curvature optimization (without IQP)
use_opt_mincurv_iqp = True          # minimum curvature optimization (with IQP)
use_opt_shortest_path = False       # shortest path optimization


# ----------------------------------------------------------------------------------------------------------------------
# CHECK PYTHON DEPENDENCIES --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# get current path
file_paths["module"] = os.path.dirname(os.path.abspath(__file__))

# read dependencies from requirements.txt
requirements_path = os.path.join(file_paths["module"], 'requirements.txt')
dependencies = []

with open(requirements_path, 'r') as fh:
    line = fh.readline()

    while line:
        dependencies.append(line.rstrip())
        line = fh.readline()

# check dependencies
pkg_resources.require(dependencies)


# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION STUFF -------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# set together track import path
file_paths["track"] = file_paths["module"] + "/inputs/tracks/" + file_paths["track_file"]

# set export paths
file_paths["racetraj_export"] = file_paths["module"] + "/outputs/racetraj_cl.csv"


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT VEHICLE DEPENDENT PARAMETERS ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# load vehicle parameter file into a "pars" dict
parser = configparser.ConfigParser()
pars = {}

if not parser.read(file_paths["module"] + "/params/" + file_paths["veh_params_file"]):
    raise ValueError('Specified config file does not exist or is empty!')

pars["ggv"] = json.loads(parser.get('GGV', 'ggv'))
pars["stepsizes"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'stepsizes'))
pars["reg_smooth_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'reg_smooth_opts'))
pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts'))
pars["veh_dims"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'veh_dims'))

# set import path for ggv diagram
file_paths["ggv"] = file_paths["module"] + "/inputs/ggv/" + pars["ggv"]


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT TRACK AND GGV DIAGRAMM ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# save start time
t_start = time.perf_counter()

reftrack_imp, ggv = process_functions.src.imp_track_ggv.imp_track_ggv(imp_opts=imp_opts,
                                                                      file_paths=file_paths,
                                                                      veh_dims=pars["veh_dims"])


# ----------------------------------------------------------------------------------------------------------------------
# PREPARE REFTRACK -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
    process_functions.src.prep_track.prep_track(reftrack_imp=reftrack_imp,
                                                pars=pars,
                                                debug=debug,
                                                check_normal_crossings=check_normal_crossings)


# ----------------------------------------------------------------------------------------------------------------------
# CALL OPTIMIZATION ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if use_opt_mincurv:
    alpha_opt = opt_geometrical.src.opt_min_curv.opt_min_curv(reftrack=reftrack_interp,
                                                              normvectors=normvec_normalized_interp,
                                                              A=a_interp,
                                                              kappa_bound=pars["optim_opts"]["kappa_bound"],
                                                              w_veh=pars["optim_opts"]["w_veh"],
                                                              print_debug=debug,
                                                              plot_debug=plot_opts["opt_min_curv"])[0]

elif use_opt_mincurv_iqp:
    alpha_opt, reftrack_interp, normvec_normalized_interp = process_functions.src.iqp_handler.\
        iqp_handler(reftrack=reftrack_interp,
                    normvectors=normvec_normalized_interp,
                    A=a_interp,
                    kappa_bound=pars["optim_opts"]["kappa_bound"],
                    w_veh=pars["optim_opts"]["w_veh"],
                    print_debug=debug,
                    plot_debug=plot_opts["opt_min_curv"],
                    stepsize_reg=pars["stepsizes"]["stepsize_reg"],
                    iters_min=pars["optim_opts"]["iqp_iters_min"],
                    curv_error_allowed=pars["optim_opts"]["iqp_curv_error_allowed"])

elif use_opt_shortest_path:
    alpha_opt = opt_geometrical.src.opt_shortest_path.opt_shortest_path(reftrack=reftrack_interp,
                                                                        normvectors=normvec_normalized_interp,
                                                                        w_veh=pars["optim_opts"]["w_veh"],
                                                                        print_debug=debug)

else:
    alpha_opt = np.zeros(reftrack_interp.shape[0])


# ----------------------------------------------------------------------------------------------------------------------
# INTERPOLATE SPLINES TO SMALL DISTANCES BETWEEN RACELINE POINTS -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = process_functions.src.interp_raceline.\
    interp_raceline(stepsize_interp_after_opt=pars["stepsizes"]["stepsize_interp_after_opt"],
                    refline_interp=reftrack_interp[:, :2],
                    alpha_opt=alpha_opt,
                    normvec_normalized_interp=normvec_normalized_interp)


# ----------------------------------------------------------------------------------------------------------------------
# CALCULATE HEADING AND CURVATURE --------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# calculate heading and curvature (analytically)
psi_vel_opt, kappa_opt = tph.calc_head_curv_an.\
    calc_head_curv_an(coeffs_x=coeffs_x_opt,
                      coeffs_y=coeffs_y_opt,
                      ind_spls=spline_inds_opt_interp,
                      t_spls=t_vals_opt_interp)


# ----------------------------------------------------------------------------------------------------------------------
# CALCULATE VELOCITY PROFILE -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

vx_profile_opt = tph.calc_vel_profile.calc_vel_profile(ggv=ggv,
                                                       kappa=kappa_opt,
                                                       el_lengths=el_lengths_opt_interp,
                                                       dyn_model_exp=pars["optim_opts"]["dyn_model_exp"],
                                                       filt_window=pars["optim_opts"]["window_size_conv_filt"],
                                                       closed=True)

# calculate longitudinal acceleration profile
vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                     el_lengths=el_lengths_opt_interp,
                                                     eq_length_output=False)

# calculate laptime
t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                 ax_profile=ax_profile_opt,
                                                 el_lengths=el_lengths_opt_interp)
print("Laptime: %.2f s" % t_profile_cl[-1])

if plot_opts["velprofile"]:
    s_points = np.cumsum(el_lengths_opt_interp[:-1])
    s_points = np.insert(s_points, 0, 0.0)

    plt.plot(s_points, vx_profile_opt)
    plt.plot(s_points, ax_profile_opt)
    plt.plot(s_points, t_profile_cl[:-1])

    plt.grid()
    plt.xlabel("distance in m")
    plt.legend(["vx in mps", "ax in mps2", "t in s"])

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# DATA POSTPROCESSING --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# arrange data into one trajectory
trajectory_opt = np.column_stack((s_points_opt_interp, raceline_interp, psi_vel_opt, kappa_opt, vx_profile_opt,
                                  ax_profile_opt))
spline_data_opt = np.column_stack((spline_lengths_opt, coeffs_x_opt, coeffs_y_opt))

# create a closed race trajectory array
trajectory_opt_cl = np.vstack((trajectory_opt, trajectory_opt[0, :]))
trajectory_opt_cl[-1, 0] = np.sum(spline_data_opt[:, 0])  # set correct length

# print end time
print("Runtime from referenceline import to trajectory export was %.2f s" % (time.perf_counter() - t_start))


# ----------------------------------------------------------------------------------------------------------------------
# CHECK TRAJECTORY -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

bound1, bound2 = process_functions.src.check_traj.\
    check_traj(reftrack=reftrack_interp,
               reftrack_normvec_normalized=normvec_normalized_interp,
               veh_dims=pars["veh_dims"],
               debug=debug,
               trajectory_opt=trajectory_opt,
               ggv=ggv)


# ----------------------------------------------------------------------------------------------------------------------
# EXPORT ---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# export data to CSV
process_functions.src.export_traj.export_traj(file_paths=file_paths,
                                              traj_race=trajectory_opt_cl)

print("\nFinished creation of trajectory:", time.strftime("%H:%M:%S"), "\n")


# ----------------------------------------------------------------------------------------------------------------------
# PLOT RESULTS ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

process_functions.src.plot_funcs.plot_funcs(plot_opts=plot_opts,
                                            optim_opts=pars["optim_opts"],
                                            veh_dims=pars["veh_dims"],
                                            refline_interp=reftrack_interp[:, :2],
                                            bound1=bound1,
                                            bound2=bound2,
                                            trajectory_opt=trajectory_opt)
