import numpy as np
import time
import json
import os
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
import configparser
import pkg_resources
import helper_funcs_glob

"""
Created by:
Alexander Heilmeier

Documentation:
This script has to be executed to generate an optimal trajectory based on a given reference track.
"""

# ----------------------------------------------------------------------------------------------------------------------
# USER INPUT -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# choose vehicle parameter file
file_paths = {"veh_params_file": "racecar.ini"}

# debug and plot options
debug = True                                    # print console messages
plot_opts = {"mincurv_curv_lin": False,         # plot curv. linearization (original and solution based) (mincurv only)
             "raceline": True,                  # plot optimized path
             "raceline_curv": True,             # plot curvature profile of optimized path
             "racetraj_vel": True,              # plot velocity profile
             "racetraj_vel_3d": False,          # plot 3D velocity profile above raceline
             "racetraj_vel_3d_stepsize": 1.0,   # [m] vertical lines stepsize in 3D velocity profile plot
             "spline_normals": False}           # plot spline normals

# select track file (including centerline coords + track widths)
file_paths["track_name"] = "berlin_2018"
# file_paths["track_name"] = "handling_track"
# file_paths["track_name"] = "rounded_rectangle"

# set import options
# berlin_2018: set_new_start 106.0, 141.0
imp_opts = {"flip_imp_track": False,                    # flip imported track to reverse direction
            "set_new_start": False,                     # set new starting point (changes order, not coordinates)
            "new_start": np.array([106.0, 141.0])}      # [x_m, y_m]

# check normal vector crossings (can take a while)
check_normal_crossings = False

# set optimization type
# 'shortest_path'       shortest path optimization
# 'mincurv'             minimum curvature optimization without iterative call
# 'mincurv_iqp'         minimum curvature optimization with iterative call
opt_type = 'mincurv_iqp'

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
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + '.csv')

# set export paths
file_paths["traj_race_export"] = file_paths["module"] + "/outputs/traj_race_cl.csv"

# create outputs folder
os.makedirs(file_paths["module"] + "/outputs", exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT VEHICLE DEPENDENT PARAMETERS ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# load vehicle parameter file into a "pars" dict
parser = configparser.ConfigParser()
pars = {}

if not parser.read(file_paths["module"] + "/params/" + file_paths["veh_params_file"]):
    raise ValueError('Specified config file does not exist or is empty!')

pars["ggv"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv'))
pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

if opt_type == 'shortest_path':
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_shortest_path'))

elif opt_type in ['mincurv', 'mincurv_iqp']:
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

else:
    raise ValueError('Unknown optimization type!')

# set import path for ggv diagram
file_paths["ggv"] = file_paths["module"] + "/inputs/ggv/" + pars["ggv"]

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT TRACK AND GGV DIAGRAMM ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# save start time
t_start = time.perf_counter()

reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                               file_path=file_paths["track_file"],
                                                               width_veh=pars["veh_params"]["width"])

ggv = tph.import_ggv.import_ggv(ggv_import_path=file_paths["ggv"])

# ----------------------------------------------------------------------------------------------------------------------
# PREPARE REFTRACK -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
    helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack_imp,
                                                reg_smooth_opts=pars["reg_smooth_opts"],
                                                stepsize_opts=pars["stepsize_opts"],
                                                debug=debug,
                                                check_normal_crossings=check_normal_crossings)

# ----------------------------------------------------------------------------------------------------------------------
# CALL OPTIMIZATION ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if opt_type == 'mincurv':
    alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack=reftrack_interp,
                                              normvectors=normvec_normalized_interp,
                                              A=a_interp,
                                              kappa_bound=pars["optim_opts"]["curvlim"],
                                              w_veh=pars["optim_opts"]["width_opt"],
                                              print_debug=debug,
                                              plot_debug=plot_opts["mincurv_curv_lin"])[0]

elif opt_type == 'mincurv_iqp':
    alpha_opt, reftrack_interp, normvec_normalized_interp = tph.iqp_handler.\
        iqp_handler(reftrack=reftrack_interp,
                    normvectors=normvec_normalized_interp,
                    A=a_interp,
                    kappa_bound=pars["optim_opts"]["curvlim"],
                    w_veh=pars["optim_opts"]["width_opt"],
                    print_debug=debug,
                    plot_debug=plot_opts["mincurv_curv_lin"],
                    stepsize_interp=pars["stepsize_opts"]["stepsize_reg"],
                    iters_min=pars["optim_opts"]["iqp_iters_min"],
                    curv_error_allowed=pars["optim_opts"]["iqp_curverror_allowed"])

elif opt_type == 'shortest_path':
    alpha_opt = tph.opt_shortest_path.opt_shortest_path(reftrack=reftrack_interp,
                                                        normvectors=normvec_normalized_interp,
                                                        w_veh=pars["optim_opts"]["width_opt"],
                                                        print_debug=debug)

else:
    alpha_opt = np.zeros(reftrack_interp.shape[0])

# ----------------------------------------------------------------------------------------------------------------------
# INTERPOLATE SPLINES TO SMALL DISTANCES BETWEEN RACELINE POINTS -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = tph.create_raceline.\
    create_raceline(refline=reftrack_interp[:, :2],
                    normvectors=normvec_normalized_interp,
                    alpha=alpha_opt,
                    stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"])

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

vx_profile_opt = tph.calc_vel_profile.\
    calc_vel_profile(ggv=ggv,
                     kappa=kappa_opt,
                     el_lengths=el_lengths_opt_interp,
                     dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
                     filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
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
print("Laptime: %.2fs" % t_profile_cl[-1])

if plot_opts["racetraj_vel"]:
    s_points = np.cumsum(el_lengths_opt_interp[:-1])
    s_points = np.insert(s_points, 0, 0.0)

    plt.plot(s_points, vx_profile_opt)
    plt.plot(s_points, ax_profile_opt)
    plt.plot(s_points, t_profile_cl[:-1])

    plt.grid()
    plt.xlabel("distance in m")
    plt.legend(["vx in m/s", "ax in m/s2", "t in s"])

    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# DATA POSTPROCESSING --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# arrange data into one trajectory
trajectory_opt = np.column_stack((s_points_opt_interp, raceline_interp, psi_vel_opt, kappa_opt, vx_profile_opt,
                                  ax_profile_opt))
spline_data_opt = np.column_stack((spline_lengths_opt, coeffs_x_opt, coeffs_y_opt))

# create a closed race trajectory array
traj_race_cl = np.vstack((trajectory_opt, trajectory_opt[0, :]))
traj_race_cl[-1, 0] = np.sum(spline_data_opt[:, 0])  # set correct length

# print end time
print("Runtime from referenceline import to trajectory export was %.2fs" % (time.perf_counter() - t_start))

# ----------------------------------------------------------------------------------------------------------------------
# CHECK TRAJECTORY -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

bound1, bound2 = helper_funcs_glob.src.check_traj.\
    check_traj(reftrack=reftrack_interp,
               reftrack_normvec_normalized=normvec_normalized_interp,
               length_veh=pars["veh_params"]["length"],
               width_veh=pars["veh_params"]["width"],
               debug=debug,
               trajectory=trajectory_opt,
               ggv=ggv,
               curvlim=pars["veh_params"]["curvlim"],
               mass_veh=pars["veh_params"]["mass"],
               dragcoeff=pars["veh_params"]["dragcoeff"])

# ----------------------------------------------------------------------------------------------------------------------
# EXPORT ---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# export data to CSV
helper_funcs_glob.src.export_traj.export_traj(file_paths=file_paths,
                                              traj_race=traj_race_cl)

print("\nFinished creation of trajectory:", time.strftime("%H:%M:%S"), "\n")

# ----------------------------------------------------------------------------------------------------------------------
# PLOT RESULTS ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

helper_funcs_glob.src.result_plots.result_plots(plot_opts=plot_opts,
                                                width_veh_opt=pars["optim_opts"]["width_opt"],
                                                width_veh_real=pars["veh_params"]["width"],
                                                refline=reftrack_interp[:, :2],
                                                bound1=bound1,
                                                bound2=bound2,
                                                trajectory=trajectory_opt)
