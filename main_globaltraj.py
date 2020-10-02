import opt_mintime_traj
import numpy as np
import time
import json
import os
import trajectory_planning_helpers as tph
import copy
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

# choose vehicle parameter file ----------------------------------------------------------------------------------------
file_paths = {"veh_params_file": "racecar.ini"}

# debug and plot options -----------------------------------------------------------------------------------------------
debug = True                                    # print console messages
plot_opts = {"mincurv_curv_lin": False,         # plot curv. linearization (original and solution based) (mincurv only)
             "raceline": True,                  # plot optimized path
             "imported_bounds": False,          # plot imported bounds (analyze difference to interpolated bounds)
             "raceline_curv": True,             # plot curvature profile of optimized path
             "racetraj_vel": True,              # plot velocity profile
             "racetraj_vel_3d": False,          # plot 3D velocity profile above raceline
             "racetraj_vel_3d_stepsize": 1.0,   # [m] vertical lines stepsize in 3D velocity profile plot
             "spline_normals": False,           # plot spline normals to check for crossings
             "mintime_plots": False}            # plot states, controls, friction coeffs etc. (mintime only)

# select track file (including centerline coordinates + track widths) --------------------------------------------------
# file_paths["track_name"] = "rounded_rectangle"                              # artificial track
# file_paths["track_name"] = "handling_track"                                 # artificial track
file_paths["track_name"] = "berlin_2018"                                    # Berlin Formula E 2018
# file_paths["track_name"] = "modena_2019"                                    # Modena 2019

# set import options ---------------------------------------------------------------------------------------------------
imp_opts = {"flip_imp_track": False,                # flip imported track to reverse direction
            "set_new_start": False,                 # set new starting point (changes order, not coordinates)
            "new_start": np.array([0.0, -47.0]),    # [x_m, y_m]
            "min_track_width": None,                # [m] minimum enforced track width (set None to deactivate)
            "num_laps": 1}                          # number of laps to be driven (significant with powertrain-option),
                                                    # only relevant in mintime-optimization

# set optimization type ------------------------------------------------------------------------------------------------
# 'shortest_path'       shortest path optimization
# 'mincurv'             minimum curvature optimization without iterative call
# 'mincurv_iqp'         minimum curvature optimization with iterative call
# 'mintime'             time-optimal trajectory optimization
opt_type = 'mintime'

# set mintime specific options (mintime only) --------------------------------------------------------------------------
# tpadata:                      set individual friction map data file if desired (e.g. for varmue maps), else set None,
#                               e.g. "berlin_2018_varmue08-12_tpadata.json"
# warm_start:                   [True/False] warm start IPOPT if previous result is available for current track
# var_friction:                 [-] None, "linear", "gauss" -> set if variable friction coefficients should be used
#                               either with linear regression or with gaussian basis functions (requires friction map)
# reopt_mintime_solution:       reoptimization of the mintime solution by min. curv. opt. for improved curv. smoothness
# recalc_vel_profile_by_tph:    override mintime velocity profile by ggv based calculation (see TPH package)

mintime_opts = {"tpadata": None,
                "warm_start": False,
                "var_friction": None,
                "reopt_mintime_solution": False,
                "recalc_vel_profile_by_tph": False}

# lap time calculation table -------------------------------------------------------------------------------------------
lap_time_mat_opts = {"use_lap_time_mat": False,             # calculate a lap time matrix (diff. top speeds and scales)
                     "gg_scale_range": [0.3, 1.0],          # range of gg scales to be covered
                     "gg_scale_stepsize": 0.05,             # step size to be applied
                     "top_speed_range": [100.0, 150.0],     # range of top speeds to be simulated [in km/h]
                     "top_speed_stepsize": 5.0,             # step size to be applied
                     "file": "lap_time_matrix.csv"}         # file name of the lap time matrix (stored in "outputs")

# ----------------------------------------------------------------------------------------------------------------------
# CHECK USER INPUT -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if opt_type not in ["shortest_path", "mincurv", "mincurv_iqp", "mintime"]:
    raise IOError("Unknown optimization type!")

if opt_type == "mintime" and not mintime_opts["recalc_vel_profile_by_tph"] and lap_time_mat_opts["use_lap_time_mat"]:
    raise IOError("Lap time calculation table should be created but velocity profile recalculation with TPH solver is"
                  " not allowed!")

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
# INITIALIZATION OF PATHS ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# assemble track import path
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")

# assemble friction map import paths
file_paths["tpamap"] = os.path.join(file_paths["module"], "inputs", "frictionmaps",
                                    file_paths["track_name"] + "_tpamap.csv")

if mintime_opts["tpadata"] is None:
    file_paths["tpadata"] = os.path.join(file_paths["module"], "inputs", "frictionmaps",
                                         file_paths["track_name"] + "_tpadata.json")
else:
    file_paths["tpadata"] = os.path.join(file_paths["module"], "inputs", "frictionmaps", mintime_opts["tpadata"])

# check if friction map files are existing if the var_friction option was set
if opt_type == 'mintime' \
        and mintime_opts["var_friction"] is not None \
        and not (os.path.exists(file_paths["tpadata"]) and os.path.exists(file_paths["tpamap"])):

    mintime_opts["var_friction"] = None
    print("WARNING: var_friction option is not None but friction map data is missing for current track -> Setting"
          " var_friction to None!")

# create outputs folder(s)
os.makedirs(file_paths["module"] + "/outputs", exist_ok=True)

if opt_type == 'mintime':
    os.makedirs(file_paths["module"] + "/outputs/mintime", exist_ok=True)

# assemble export paths
file_paths["mintime_export"] = os.path.join(file_paths["module"], "outputs", "mintime")
file_paths["traj_race_export"] = os.path.join(file_paths["module"], "outputs", "traj_race_cl.csv")
# file_paths["traj_ltpl_export"] = os.path.join(file_paths["module"], "outputs", "traj_ltpl_cl.csv")
file_paths["lap_time_mat_export"] = os.path.join(file_paths["module"], "outputs", lap_time_mat_opts["file"])

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT VEHICLE DEPENDENT PARAMETERS ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# load vehicle parameter file into a "pars" dict
parser = configparser.ConfigParser()
pars = {}

if not parser.read(os.path.join(file_paths["module"], "params", file_paths["veh_params_file"])):
    raise ValueError('Specified config file does not exist or is empty!')

pars["ggv_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
pars["ax_max_machines_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

if opt_type == 'shortest_path':
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_shortest_path'))

elif opt_type in ['mincurv', 'mincurv_iqp']:
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

elif opt_type == 'mintime':
    pars["curv_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'curv_calc_opts'))
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
    pars["vehicle_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
    pars["tire_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'tire_params_mintime'))
    pars["pwr_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'pwr_params_mintime'))

    # modification of mintime options/parameters
    pars["optim_opts"]["var_friction"] = mintime_opts["var_friction"]
    pars["optim_opts"]["warm_start"] = mintime_opts["warm_start"]
    pars["vehicle_params_mintime"]["wheelbase"] = (pars["vehicle_params_mintime"]["wheelbase_front"]
                                                   + pars["vehicle_params_mintime"]["wheelbase_rear"])

# set import path for ggv diagram and ax_max_machines (if required)
if not (opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]):
    file_paths["ggv_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info", pars["ggv_file"])
    file_paths["ax_max_machines_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info",
                                                      pars["ax_max_machines_file"])

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT TRACK AND VEHICLE DYNAMICS INFORMATION ------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# save start time
t_start = time.perf_counter()

# import track
reftrack_imp = helper_funcs_glob.src.import_track.import_track(imp_opts=imp_opts,
                                                               file_path=file_paths["track_file"],
                                                               width_veh=pars["veh_params"]["width"])

# import ggv and ax_max_machines (if required)
if not (opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]):
    ggv, ax_max_machines = tph.import_veh_dyn_info.\
        import_veh_dyn_info(ggv_import_path=file_paths["ggv_file"],
                            ax_max_machines_import_path=file_paths["ax_max_machines_file"])
else:
    ggv = None
    ax_max_machines = None

# set ax_pos_safe / ax_neg_safe / ay_safe if required and not set in parameters file
if opt_type == 'mintime' and pars["optim_opts"]["safe_traj"] \
        and (pars["optim_opts"]["ax_pos_safe"] is None
             or pars["optim_opts"]["ax_neg_safe"] is None
             or pars["optim_opts"]["ay_safe"] is None):

    # get ggv if not available
    if ggv is None:
        ggv = tph.import_veh_dyn_info. \
            import_veh_dyn_info(ggv_import_path=file_paths["ggv_file"],
                                ax_max_machines_import_path=file_paths["ax_max_machines_file"])[0]

    # limit accelerations
    if pars["optim_opts"]["ax_pos_safe"] is None:
        pars["optim_opts"]["ax_pos_safe"] = np.amin(ggv[:, 1])
    if pars["optim_opts"]["ax_neg_safe"] is None:
        pars["optim_opts"]["ax_neg_safe"] = -np.amin(ggv[:, 1])
    if pars["optim_opts"]["ay_safe"] is None:
        pars["optim_opts"]["ay_safe"] = np.amin(ggv[:, 2])

# ----------------------------------------------------------------------------------------------------------------------
# PREPARE REFTRACK -----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
    helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=reftrack_imp,
                                                reg_smooth_opts=pars["reg_smooth_opts"],
                                                stepsize_opts=pars["stepsize_opts"],
                                                debug=debug,
                                                min_width=imp_opts["min_track_width"])

# ----------------------------------------------------------------------------------------------------------------------
# CALL OPTIMIZATION ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# if reoptimization of mintime solution is used afterwards we have to consider some additional deviation in the first
# optimization
if opt_type == 'mintime' and mintime_opts["reopt_mintime_solution"]:
    w_veh_tmp = pars["optim_opts"]["width_opt"] + (pars["optim_opts"]["w_tr_reopt"] - pars["optim_opts"]["w_veh_reopt"])
    w_veh_tmp += pars["optim_opts"]["w_add_spl_regr"]
    pars_tmp = copy.deepcopy(pars)
    pars_tmp["optim_opts"]["width_opt"] = w_veh_tmp
else:
    pars_tmp = pars

# call optimization
if opt_type == 'mincurv':
    alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack=reftrack_interp,
                                              normvectors=normvec_normalized_interp,
                                              A=a_interp,
                                              kappa_bound=pars["veh_params"]["curvlim"],
                                              w_veh=pars["optim_opts"]["width_opt"],
                                              print_debug=debug,
                                              plot_debug=plot_opts["mincurv_curv_lin"])[0]

elif opt_type == 'mincurv_iqp':
    alpha_opt, reftrack_interp, normvec_normalized_interp = tph.iqp_handler.\
        iqp_handler(reftrack=reftrack_interp,
                    normvectors=normvec_normalized_interp,
                    A=a_interp,
                    kappa_bound=pars["veh_params"]["curvlim"],
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

elif opt_type == 'mintime':
    # reftrack_interp, a_interp and normvec_normalized_interp are returned for the case that non-regular sampling was
    # applied
    alpha_opt, v_opt, reftrack_interp, a_interp_tmp, normvec_normalized_interp = opt_mintime_traj.src.opt_mintime.\
        opt_mintime(reftrack=reftrack_interp,
                    coeffs_x=coeffs_x_interp,
                    coeffs_y=coeffs_y_interp,
                    normvectors=normvec_normalized_interp,
                    pars=pars_tmp,
                    tpamap_path=file_paths["tpamap"],
                    tpadata_path=file_paths["tpadata"],
                    export_path=file_paths["mintime_export"],
                    print_debug=debug,
                    plot_debug=plot_opts["mintime_plots"])

    # replace a_interp if necessary
    if a_interp_tmp is not None:
        a_interp = a_interp_tmp

else:
    raise ValueError('Unknown optimization type!')
    # alpha_opt = np.zeros(reftrack_interp.shape[0])

# ----------------------------------------------------------------------------------------------------------------------
# REOPTIMIZATION OF THE MINTIME SOLUTION -------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if opt_type == 'mintime' and mintime_opts["reopt_mintime_solution"]:

    # get raceline solution of the time-optimal trajectory
    raceline_mintime = reftrack_interp[:, :2] + np.expand_dims(alpha_opt, 1) * normvec_normalized_interp

    # calculate new track boundaries around raceline solution depending on alpha_opt values
    w_tr_right_mintime = reftrack_interp[:, 2] - alpha_opt
    w_tr_left_mintime = reftrack_interp[:, 3] + alpha_opt

    # create new reference track around the raceline
    racetrack_mintime = np.column_stack((raceline_mintime, w_tr_right_mintime, w_tr_left_mintime))

    # use spline approximation a second time
    reftrack_interp, normvec_normalized_interp, a_interp = \
        helper_funcs_glob.src.prep_track.prep_track(reftrack_imp=racetrack_mintime,
                                                    reg_smooth_opts=pars["reg_smooth_opts"],
                                                    stepsize_opts=pars["stepsize_opts"],
                                                    debug=False,
                                                    min_width=imp_opts["min_track_width"])[:3]

    # set artificial track widths for reoptimization
    w_tr_tmp = 0.5 * pars["optim_opts"]["w_tr_reopt"] * np.ones(reftrack_interp.shape[0])
    racetrack_mintime_reopt = np.column_stack((reftrack_interp[:, :2], w_tr_tmp, w_tr_tmp))

    # call mincurv reoptimization
    alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack=racetrack_mintime_reopt,
                                              normvectors=normvec_normalized_interp,
                                              A=a_interp,
                                              kappa_bound=pars["veh_params"]["curvlim"],
                                              w_veh=pars["optim_opts"]["w_veh_reopt"],
                                              print_debug=debug,
                                              plot_debug=plot_opts["mincurv_curv_lin"])[0]

    # calculate minimum distance from raceline to bounds and print it
    if debug:
        raceline_reopt = reftrack_interp[:, :2] + np.expand_dims(alpha_opt, 1) * normvec_normalized_interp
        bound_r_reopt = (reftrack_interp[:, :2]
                         + np.expand_dims(reftrack_interp[:, 2], axis=1) * normvec_normalized_interp)
        bound_l_reopt = (reftrack_interp[:, :2]
                         - np.expand_dims(reftrack_interp[:, 3], axis=1) * normvec_normalized_interp)

        d_r_reopt = np.hypot(raceline_reopt[:, 0] - bound_r_reopt[:, 0], raceline_reopt[:, 1] - bound_r_reopt[:, 1])
        d_l_reopt = np.hypot(raceline_reopt[:, 0] - bound_l_reopt[:, 0], raceline_reopt[:, 1] - bound_l_reopt[:, 1])

        print("INFO: Mintime reoptimization: minimum distance to right/left bound: %.2fm / %.2fm"
              % (np.amin(d_r_reopt) - pars["veh_params"]["width"] / 2,
                 np.amin(d_l_reopt) - pars["veh_params"]["width"] / 2))

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
# CALCULATE VELOCITY AND ACCELERATION PROFILE --------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if opt_type == 'mintime' and not mintime_opts["recalc_vel_profile_by_tph"]:
    # interpolation
    s_splines = np.cumsum(spline_lengths_opt)
    s_splines = np.insert(s_splines, 0, 0.0)
    vx_profile_opt = np.interp(s_points_opt_interp, s_splines[:-1], v_opt)

else:
    vx_profile_opt = tph.calc_vel_profile.\
        calc_vel_profile(ggv=ggv,
                         ax_max_machines=ax_max_machines,
                         v_max=pars["veh_params"]["v_max"],
                         kappa=kappa_opt,
                         el_lengths=el_lengths_opt_interp,
                         closed=True,
                         filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
                         dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
                         drag_coeff=pars["veh_params"]["dragcoeff"],
                         m_veh=pars["veh_params"]["mass"])

# calculate longitudinal acceleration profile
vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                     el_lengths=el_lengths_opt_interp,
                                                     eq_length_output=False)

# calculate laptime
t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                 ax_profile=ax_profile_opt,
                                                 el_lengths=el_lengths_opt_interp)
print("INFO: Estimated laptime: %.2fs" % t_profile_cl[-1])

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
# CALCULATE LAP TIMES (AT DIFFERENT SCALES AND TOP SPEEDS) -------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if lap_time_mat_opts["use_lap_time_mat"]:
    # simulate lap times
    ggv_scales = np.linspace(lap_time_mat_opts['gg_scale_range'][0],
                             lap_time_mat_opts['gg_scale_range'][1],
                             int((lap_time_mat_opts['gg_scale_range'][1] - lap_time_mat_opts['gg_scale_range'][0])
                                 / lap_time_mat_opts['gg_scale_stepsize']) + 1)
    top_speeds = np.linspace(lap_time_mat_opts['top_speed_range'][0] / 3.6,
                             lap_time_mat_opts['top_speed_range'][1] / 3.6,
                             int((lap_time_mat_opts['top_speed_range'][1] - lap_time_mat_opts['top_speed_range'][0])
                                 / lap_time_mat_opts['top_speed_stepsize']) + 1)

    # setup results matrix
    lap_time_matrix = np.zeros((top_speeds.shape[0] + 1, ggv_scales.shape[0] + 1))

    # write parameters in first column and row
    lap_time_matrix[1:, 0] = top_speeds * 3.6
    lap_time_matrix[0, 1:] = ggv_scales

    for i, top_speed in enumerate(top_speeds):
        for j, ggv_scale in enumerate(ggv_scales):
            tph.progressbar.progressbar(i*ggv_scales.shape[0] + j,
                                        top_speeds.shape[0] * ggv_scales.shape[0],
                                        prefix="Simulating laptimes ")

            ggv_mod = np.copy(ggv)
            ggv_mod[:, 1:] *= ggv_scale

            vx_profile_opt = tph.calc_vel_profile.\
                calc_vel_profile(ggv=ggv_mod,
                                 ax_max_machines=ax_max_machines,
                                 v_max=top_speed,
                                 kappa=kappa_opt,
                                 el_lengths=el_lengths_opt_interp,
                                 dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
                                 filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
                                 closed=True,
                                 drag_coeff=pars["veh_params"]["dragcoeff"],
                                 m_veh=pars["veh_params"]["mass"])

            # calculate longitudinal acceleration profile
            vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
            ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                                 el_lengths=el_lengths_opt_interp,
                                                                 eq_length_output=False)

            # calculate lap time
            t_profile_cl = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                                             ax_profile=ax_profile_opt,
                                                             el_lengths=el_lengths_opt_interp)

            # store entry in lap time matrix
            lap_time_matrix[i + 1, j + 1] = t_profile_cl[-1]

    # store lap time matrix to file
    np.savetxt(file_paths["lap_time_mat_export"], lap_time_matrix, delimiter=",", fmt="%.3f")

# ----------------------------------------------------------------------------------------------------------------------
# DATA POSTPROCESSING --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# arrange data into one trajectory
trajectory_opt = np.column_stack((s_points_opt_interp,
                                  raceline_interp,
                                  psi_vel_opt,
                                  kappa_opt,
                                  vx_profile_opt,
                                  ax_profile_opt))
spline_data_opt = np.column_stack((spline_lengths_opt, coeffs_x_opt, coeffs_y_opt))

# create a closed race trajectory array
traj_race_cl = np.vstack((trajectory_opt, trajectory_opt[0, :]))
traj_race_cl[-1, 0] = np.sum(spline_data_opt[:, 0])  # set correct length

# print end time
print("INFO: Runtime from import to final trajectory was %.2fs" % (time.perf_counter() - t_start))

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
               ax_max_machines=ax_max_machines,
               v_max=pars["veh_params"]["v_max"],
               curvlim=pars["veh_params"]["curvlim"],
               mass_veh=pars["veh_params"]["mass"],
               dragcoeff=pars["veh_params"]["dragcoeff"])

# ----------------------------------------------------------------------------------------------------------------------
# EXPORT ---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# export race trajectory  to CSV
if "traj_race_export" in file_paths.keys():
    helper_funcs_glob.src.export_traj_race.export_traj_race(file_paths=file_paths,
                                                            traj_race=traj_race_cl)

# if requested, export trajectory including map information (via normal vectors) to CSV
if "traj_ltpl_export" in file_paths.keys():
    helper_funcs_glob.src.export_traj_ltpl.export_traj_ltpl(file_paths=file_paths,
                                                            spline_lengths_opt=spline_lengths_opt,
                                                            trajectory_opt=trajectory_opt,
                                                            reftrack=reftrack_interp,
                                                            normvec_normalized=normvec_normalized_interp,
                                                            alpha_opt=alpha_opt)

print("INFO: Finished export of trajectory:", time.strftime("%H:%M:%S"))

# ----------------------------------------------------------------------------------------------------------------------
# PLOT RESULTS ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# get bound of imported map (for reference in final plot)
bound1_imp = None
bound2_imp = None

if plot_opts["imported_bounds"]:
    # try to extract four times as many points as in the interpolated version (in order to hold more details)
    n_skip = max(int(reftrack_imp.shape[0] / (bound1.shape[0] * 4)), 1)

    _, _, _, normvec_imp = tph.calc_splines.calc_splines(path=np.vstack((reftrack_imp[::n_skip, 0:2],
                                                                         reftrack_imp[0, 0:2])))

    bound1_imp = reftrack_imp[::n_skip, :2] + normvec_imp * np.expand_dims(reftrack_imp[::n_skip, 2], 1)
    bound2_imp = reftrack_imp[::n_skip, :2] - normvec_imp * np.expand_dims(reftrack_imp[::n_skip, 3], 1)

# plot results
helper_funcs_glob.src.result_plots.result_plots(plot_opts=plot_opts,
                                                width_veh_opt=pars["optim_opts"]["width_opt"],
                                                width_veh_real=pars["veh_params"]["width"],
                                                refline=reftrack_interp[:, :2],
                                                bound1_imp=bound1_imp,
                                                bound2_imp=bound2_imp,
                                                bound1_interp=bound1,
                                                bound2_interp=bound2,
                                                trajectory=trajectory_opt)
