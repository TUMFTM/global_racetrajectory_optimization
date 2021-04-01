import os
import sys
import time
import numpy as np
import casadi as ca
import opt_mintime_traj
import trajectory_planning_helpers as tph


def opt_mintime(reftrack: np.ndarray,
                coeffs_x: np.ndarray,
                coeffs_y: np.ndarray,
                normvectors: np.ndarray,
                pars: dict,
                tpamap_path: str,
                tpadata_path: str,
                export_path: str,
                print_debug: bool = False,
                plot_debug: bool = False) -> tuple:
    """
    Created by:
    Fabian Christ

    Extended by:
    Thomas Herrmann, Francesco Passigato

    Documentation:
    The minimum lap time problem is described as an optimal control problem, converted to a nonlinear program using
    direct orthogonal Gauss-Legendre collocation and then solved by the interior-point method IPOPT. Reduced computing
    times are achieved using a curvilinear abscissa approach for track description, algorithmic differentiation using
    the software framework CasADi, and a smoothing of the track input data by approximate spline regression. The
    vehicles behavior is approximated as a double track model with quasi-steady state tire load simplification and
    nonlinear tire model.

    Please refer to our paper for further information:
    Christ, Wischnewski, Heilmeier, Lohmann
    Time-Optimal Trajectory Planning for a Race Car Considering Variable Tire-Road Friction Coefficients

    Inputs:
    reftrack:       track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    coeffs_x:       coefficient matrix of the x splines with size (no_splines x 4)
    coeffs_y:       coefficient matrix of the y splines with size (no_splines x 4)
    normvectors:    array containing normalized normal vectors for every traj. point [x_component, y_component]
    pars:           parameters dictionary
    tpamap_path:    file path to tpa map (required for friction map loading)
    tpadata_path:   file path to tpa data (required for friction map loading)
    export_path:    path to output folder for warm start files and solution files
    print_debug:    determines if debug messages are printed
    plot_debug:     determines if debug plots are shown

    Outputs:
    alpha_opt:      solution vector of the optimization problem containing the lateral shift in m for every point
    v_opt:          velocity profile for the raceline
    reftrack:       possibly (depending on non-regular sampling) modified reference track must be returned for later
                    usage
    a_interp:       possibly (depending on non-regular sampling) modified equation system matrix for splines must be
                    returned for later usage
    normvectors:    possibly (depending on non-regular sampling) modified normal vectors must be returned for later
                    usage
    """

    # ------------------------------------------------------------------------------------------------------------------
    # USE NON-REGULAR SAMPLING -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points_orig = reftrack.shape[0]

    if pars["optim_opts"]["step_non_reg"] > 0:
        reftrack, discr_points = tph.nonreg_sampling.nonreg_sampling(track=reftrack,
                                                                     eps_kappa=pars["optim_opts"]["eps_kappa"],
                                                                     step_non_reg=pars["optim_opts"]["step_non_reg"])

        # relcalculate splines
        refpath_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))
        coeffs_x, coeffs_y, a_interp, normvectors = tph.calc_splines.calc_splines(path=refpath_cl)

    else:
        discr_points = np.arange(reftrack.shape[0])
        a_interp = None

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE TRACK INFORMATION ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # spline lengths
    spline_lengths_refline = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x,
                                                                         coeffs_y=coeffs_y)

    # calculate heading and curvature (numerically)
    kappa_refline = tph.calc_head_curv_num. \
        calc_head_curv_num(path=reftrack[:, :2],
                           el_lengths=spline_lengths_refline,
                           is_closed=True,
                           stepsize_curv_preview=pars["curv_calc_opts"]["d_preview_curv"],
                           stepsize_curv_review=pars["curv_calc_opts"]["d_review_curv"],
                           stepsize_psi_preview=pars["curv_calc_opts"]["d_preview_head"],
                           stepsize_psi_review=pars["curv_calc_opts"]["d_review_head"])[1]

    # close track
    kappa_refline_cl = np.append(kappa_refline, kappa_refline[0])
    discr_points_cl = np.append(discr_points, no_points_orig)  # add virtual index of last/first point for closed track
    w_tr_left_cl = np.append(reftrack[:, 3], reftrack[0, 3])
    w_tr_right_cl = np.append(reftrack[:, 2], reftrack[0, 2])

    # step size along the reference line
    h = pars["stepsize_opts"]["stepsize_reg"]

    # optimization steps (0, 1, 2 ... end point/start point)
    # steps = [i for i in range(kappa_refline_cl.size)]
    steps = [i for i in range(discr_points_cl.size)]

    # number of control intervals
    N = steps[-1]

    # station along the reference line
    # s_opt = np.linspace(0.0, N * h, N + 1)
    s_opt = np.asarray(discr_points_cl) * h

    # interpolate curvature of reference line in terms of steps
    kappa_interp = ca.interpolant('kappa_interp', 'linear', [steps], kappa_refline_cl)

    # interpolate track width (left and right to reference line) in terms of steps
    w_tr_left_interp = ca.interpolant('w_tr_left_interp', 'linear', [steps], w_tr_left_cl)
    w_tr_right_interp = ca.interpolant('w_tr_right_interp', 'linear', [steps], w_tr_right_cl)

    # describe friction coefficients from friction map with linear equations or gaussian basis functions
    if pars["optim_opts"]["var_friction"] is not None:
        w_mue_fl, w_mue_fr, w_mue_rl, w_mue_rr, center_dist = opt_mintime_traj.src. \
            approx_friction_map.approx_friction_map(reftrack=reftrack,
                                                    normvectors=normvectors,
                                                    tpamap_path=tpamap_path,
                                                    tpadata_path=tpadata_path,
                                                    pars=pars,
                                                    dn=pars["optim_opts"]["dn"],
                                                    n_gauss=pars["optim_opts"]["n_gauss"],
                                                    print_debug=print_debug,
                                                    plot_debug=plot_debug)

    # ------------------------------------------------------------------------------------------------------------------
    # DIRECT GAUSS-LEGENDRE COLLOCATION --------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # degree of interpolating polynomial
    d = 3

    # legendre collocation points
    tau = np.append(0, ca.collocation_points(d, 'legendre'))

    # coefficient matrix for formulating the collocation equation
    C = np.zeros((d + 1, d + 1))

    # coefficient matrix for formulating the collocation equation
    D = np.zeros(d + 1)

    # coefficient matrix for formulating the collocation equation
    B = np.zeros(d + 1)

    # construct polynomial basis
    for j in range(d + 1):
        # construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        # evaluate polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # evaluate time derivative of polynomial at collocation points to get the coefficients of continuity equation
        p_der = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = p_der(tau[r])

        # evaluate integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # ------------------------------------------------------------------------------------------------------------------
    # STATE VARIABLES --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # number of state variables
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        nx = 11
        nx_pwr = 6
    else:
        nx = 5
        nx_pwr = 0

    # velocity [m/s]
    v_n = ca.SX.sym('v_n')
    v_s = 50
    v = v_s * v_n

    # side slip angle [rad]
    beta_n = ca.SX.sym('beta_n')
    beta_s = 0.5
    beta = beta_s * beta_n

    # yaw rate [rad/s]
    omega_z_n = ca.SX.sym('omega_z_n')
    omega_z_s = 1
    omega_z = omega_z_s * omega_z_n

    # lateral distance to reference line (positive = left) [m]
    n_n = ca.SX.sym('n_n')
    n_s = 5.0
    n = n_s * n_n

    # relative angle to tangent on reference line [rad]
    xi_n = ca.SX.sym('xi_n')
    xi_s = 1.0
    xi = xi_s * xi_n

    if pars["pwr_params_mintime"]["pwr_behavior"]:

        # Initialize e-machine object
        machine = opt_mintime_traj.powertrain_src.src.EMachine.EMachineModel(pwr_pars=pars["pwr_params_mintime"])

        # Initialize battery object
        batt = opt_mintime_traj.powertrain_src.src.Battery.BattModel(pwr_pars=pars["pwr_params_mintime"])

        # Initialize inverter object
        inverter = opt_mintime_traj.powertrain_src.src.Inverter.InverterModel(pwr_pars=pars["pwr_params_mintime"])

        # Initialize radiator objects (2 in total)
        radiators = opt_mintime_traj.powertrain_src.src.Radiators.RadiatorModel(pwr_pars=pars["pwr_params_mintime"])

        # scaling factors for state variables
        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s,
                        machine.temp_mot_s, batt.temp_batt_s, inverter.temp_inv_s,
                        radiators.temp_cool_mi_s, radiators.temp_cool_b_s, batt.soc_batt_s])

        # put all states together
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n,
                       machine.temp_mot_n, batt.temp_batt_n, inverter.temp_inv_n,
                       radiators.temp_cool_mi_n, radiators.temp_cool_b_n, batt.soc_batt_n)

    else:

        # scaling factors for state variables
        x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s])

        # put all states together
        x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n)

    # ------------------------------------------------------------------------------------------------------------------
    # CONTROL VARIABLES ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # number of control variables
    nu = 4

    # steer angle [rad]
    delta_n = ca.SX.sym('delta_n')
    delta_s = 0.5
    delta = delta_s * delta_n

    # positive longitudinal force (drive) [N]
    f_drive_n = ca.SX.sym('f_drive_n')
    f_drive_s = 7500.0
    f_drive = f_drive_s * f_drive_n

    # negative longitudinal force (brake) [N]
    f_brake_n = ca.SX.sym('f_brake_n')
    f_brake_s = 20000.0
    f_brake = f_brake_s * f_brake_n

    # lateral wheel load transfer [N]
    gamma_y_n = ca.SX.sym('gamma_y_n')
    gamma_y_s = 5000.0
    gamma_y = gamma_y_s * gamma_y_n

    # scaling factors for control variables
    u_s = np.array([delta_s, f_drive_s, f_brake_s, gamma_y_s])

    # put all controls together
    u = ca.vertcat(delta_n, f_drive_n, f_brake_n, gamma_y_n)

    # ------------------------------------------------------------------------------------------------------------------
    # MODEL EQUATIONS --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # extract vehicle and tire parameters
    veh = pars["vehicle_params_mintime"]
    tire = pars["tire_params_mintime"]

    # general constants
    g = pars["veh_params"]["g"]
    mass = pars["veh_params"]["mass"]

    # curvature of reference line [rad/m]
    kappa = ca.SX.sym('kappa')

    # drag force [N]
    f_xdrag = pars["veh_params"]["dragcoeff"] * v ** 2

    # rolling resistance forces [N]
    f_xroll_fl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_xroll_fr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_xroll_rl = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_xroll_rr = 0.5 * tire["c_roll"] * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_xroll = tire["c_roll"] * mass * g

    # static normal tire forces [N]
    f_zstat_fl = 0.5 * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_zstat_fr = 0.5 * mass * g * veh["wheelbase_rear"] / veh["wheelbase"]
    f_zstat_rl = 0.5 * mass * g * veh["wheelbase_front"] / veh["wheelbase"]
    f_zstat_rr = 0.5 * mass * g * veh["wheelbase_front"] / veh["wheelbase"]

    # dynamic normal tire forces (aerodynamic downforces) [N]
    f_zlift_fl = 0.5 * veh["liftcoeff_front"] * v ** 2
    f_zlift_fr = 0.5 * veh["liftcoeff_front"] * v ** 2
    f_zlift_rl = 0.5 * veh["liftcoeff_rear"] * v ** 2
    f_zlift_rr = 0.5 * veh["liftcoeff_rear"] * v ** 2

    # dynamic normal tire forces (load transfers) [N]
    f_zdyn_fl = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                 - veh["k_roll"] * gamma_y)
    f_zdyn_fr = (-0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                 + veh["k_roll"] * gamma_y)
    f_zdyn_rl = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                 - (1.0 - veh["k_roll"]) * gamma_y)
    f_zdyn_rr = (0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xdrag - f_xroll)
                 + (1.0 - veh["k_roll"]) * gamma_y)

    # sum of all normal tire forces [N]
    f_z_fl = f_zstat_fl + f_zlift_fl + f_zdyn_fl
    f_z_fr = f_zstat_fr + f_zlift_fr + f_zdyn_fr
    f_z_rl = f_zstat_rl + f_zlift_rl + f_zdyn_rl
    f_z_rr = f_zstat_rr + f_zlift_rr + f_zdyn_rr

    # slip angles [rad]
    alpha_fl = delta - ca.atan((v * ca.sin(beta) + veh["wheelbase_front"] * omega_z) /
                               (v * ca.cos(beta) - 0.5 * veh["track_width_front"] * omega_z))
    alpha_fr = delta - ca.atan((v * ca.sin(beta) + veh["wheelbase_front"] * omega_z) /
                               (v * ca.cos(beta) + 0.5 * veh["track_width_front"] * omega_z))
    alpha_rl = ca.atan((-v * ca.sin(beta) + veh["wheelbase_rear"] * omega_z) /
                       (v * ca.cos(beta) - 0.5 * veh["track_width_rear"] * omega_z))
    alpha_rr = ca.atan((-v * ca.sin(beta) + veh["wheelbase_rear"] * omega_z) /
                       (v * ca.cos(beta) + 0.5 * veh["track_width_rear"] * omega_z))

    # lateral tire forces [N]
    f_y_fl = (pars["optim_opts"]["mue"] * f_z_fl * (1 + tire["eps_front"] * f_z_fl / tire["f_z0"])
              * ca.sin(tire["C_front"] * ca.atan(tire["B_front"] * alpha_fl - tire["E_front"]
                                                 * (tire["B_front"] * alpha_fl - ca.atan(tire["B_front"] * alpha_fl)))))
    f_y_fr = (pars["optim_opts"]["mue"] * f_z_fr * (1 + tire["eps_front"] * f_z_fr / tire["f_z0"])
              * ca.sin(tire["C_front"] * ca.atan(tire["B_front"] * alpha_fr - tire["E_front"]
                                                 * (tire["B_front"] * alpha_fr - ca.atan(tire["B_front"] * alpha_fr)))))
    f_y_rl = (pars["optim_opts"]["mue"] * f_z_rl * (1 + tire["eps_rear"] * f_z_rl / tire["f_z0"])
              * ca.sin(tire["C_rear"] * ca.atan(tire["B_rear"] * alpha_rl - tire["E_rear"]
                                                * (tire["B_rear"] * alpha_rl - ca.atan(tire["B_rear"] * alpha_rl)))))
    f_y_rr = (pars["optim_opts"]["mue"] * f_z_rr * (1 + tire["eps_rear"] * f_z_rr / tire["f_z0"])
              * ca.sin(tire["C_rear"] * ca.atan(tire["B_rear"] * alpha_rr - tire["E_rear"]
                                                * (tire["B_rear"] * alpha_rr - ca.atan(tire["B_rear"] * alpha_rr)))))

    # longitudinal tire forces [N]
    f_x_fl = 0.5 * f_drive * veh["k_drive_front"] + 0.5 * f_brake * veh["k_brake_front"] - f_xroll_fl
    f_x_fr = 0.5 * f_drive * veh["k_drive_front"] + 0.5 * f_brake * veh["k_brake_front"] - f_xroll_fr
    f_x_rl = 0.5 * f_drive * (1 - veh["k_drive_front"]) + 0.5 * f_brake * (1 - veh["k_brake_front"]) - f_xroll_rl
    f_x_rr = 0.5 * f_drive * (1 - veh["k_drive_front"]) + 0.5 * f_brake * (1 - veh["k_brake_front"]) - f_xroll_rr

    # longitudinal acceleration [m/s²]
    ax = (f_x_rl + f_x_rr + (f_x_fl + f_x_fr) * ca.cos(delta) - (f_y_fl + f_y_fr) * ca.sin(delta)
          - pars["veh_params"]["dragcoeff"] * v ** 2) / mass

    # lateral acceleration [m/s²]
    ay = ((f_x_fl + f_x_fr) * ca.sin(delta) + f_y_rl + f_y_rr + (f_y_fl + f_y_fr) * ca.cos(delta)) / mass

    # ------------------------------------------------------------------------------------------------------------------
    # POWERTRAIN BEHAVIOR ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if pars["pwr_params_mintime"]["pwr_behavior"]:

        pwr_pars = pars["pwr_params_mintime"]

        # --------------------------------------------------------------------------------------------------------------
        # CALCS --------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # On wheels requested power [kW]
        p_des = (f_drive * v * 0.001)

        # E-Machines
        machine.get_states(f_drive=f_drive,
                           v=v)

        # Machine losses [kW]
        machine.get_loss(p_wheel=p_des)

        # Calculate total power loss for all electric machines in vehicle [kW]
        machine.get_machines_cum_losses()

        # Inverter losses
        inverter.get_loss(i_eff=machine.i_eff,
                          v_dc=batt.v_dc,
                          p_out_inv=machine.p_input)

        # Calculate total power loss for all inverters in vehicle [kW]
        inverter.get_inverters_cum_losses()

        # Get internal battery resistance [Ohm]
        batt.internal_resistance()

        # Get battery loss power [kW], output power [kW] and output current [A]
        batt.battery_loss(p_des=p_des,
                          p_loss_mot=machine.p_loss_total_all_machines,
                          p_loss_inv=inverter.p_loss_total_all_inverters,
                          p_in_inv=inverter.p_in_inv)

        # get intermediate temperatures for motor-inverter cooling
        radiators.get_intermediate_temps(temp_inv=inverter.temp_inv,
                                         r_inv=inverter.r_inv)

    # ------------------------------------------------------------------------------------------------------------------
    # DERIVATIVES ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # time-distance scaling factor (dt/ds)
    sf = (1.0 - n * kappa) / (v * (ca.cos(xi + beta)))

    # model equations for two track model (ordinary differential equations)
    dv = (sf / mass) * ((f_x_rl + f_x_rr) * ca.cos(beta) + (f_x_fl + f_x_fr) * ca.cos(delta - beta)
                        + (f_y_rl + f_y_rr) * ca.sin(beta) - (f_y_fl + f_y_fr) * ca.sin(delta - beta)
                        - f_xdrag * ca.cos(beta))

    dbeta = sf * (-omega_z + (-(f_x_rl + f_x_rr) * ca.sin(beta) + (f_x_fl + f_x_fr) * ca.sin(delta - beta)
                              + (f_y_rl + f_y_rr) * ca.cos(beta) + (f_y_fl + f_y_fr) * ca.cos(delta - beta)
                              + f_xdrag * ca.sin(beta)) / (mass * v))

    domega_z = (sf / veh["I_z"]) * ((f_x_rr - f_x_rl) * veh["track_width_rear"] / 2
                                    - (f_y_rl + f_y_rr) * veh["wheelbase_rear"]
                                    + ((f_x_fr - f_x_fl) * ca.cos(delta)
                                       + (f_y_fl - f_y_fr) * ca.sin(delta)) * veh["track_width_front"] / 2
                                    + ((f_y_fl + f_y_fr) * ca.cos(delta)
                                       + (f_x_fl + f_x_fr) * ca.sin(delta)) * veh["track_width_front"])

    dn = sf * v * ca.sin(xi + beta)

    dxi = sf * omega_z - kappa

    if pars["pwr_params_mintime"]["pwr_behavior"]:

        machine.get_increment(sf=sf,
                              temp_cool_12=radiators.temp_cool_12,
                              temp_cool_13=radiators.temp_cool_13)

        inverter.get_increment(sf=sf,
                               temp_cool_mi=radiators.temp_cool_mi,
                               temp_cool_12=radiators.temp_cool_12)

        batt.get_increment(sf=sf,
                           temp_cool_b=radiators.temp_cool_b)

        radiators.get_increment_mi(sf=sf,
                                   temp_mot=machine.temp_mot,
                                   temp_inv=inverter.temp_inv,
                                   r_inv=inverter.r_inv,
                                   r_machine=machine.r_machine)

        radiators.get_increment_b(sf=sf,
                                  temp_batt=batt.temp_batt,
                                  temp_cool_b=radiators.temp_cool_b,
                                  R_eq_B_inv=batt.r_batt_inverse)

        batt.get_soc(sf=sf)

        # ODEs: driving dynamics and thermodynamics
        dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi,
                        machine.dtemp, batt.dtemp, inverter.dtemp,
                        radiators.dtemp_cool_mi, radiators.dtemp_cool_b, batt.dsoc) / x_s
    else:

        # ODEs: driving dynamics only
        dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi) / x_s

    # ------------------------------------------------------------------------------------------------------------------
    # CONTROL BOUNDARIES -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    delta_min = -veh["delta_max"] / delta_s         # min. steer angle [rad]
    delta_max = veh["delta_max"] / delta_s          # max. steer angle [rad]
    f_drive_min = 0.0                               # min. longitudinal drive force [N]
    f_drive_max = veh["f_drive_max"] / f_drive_s    # max. longitudinal drive force [N]
    f_brake_min = -veh["f_brake_max"] / f_brake_s   # min. longitudinal brake force [N]
    f_brake_max = 0.0                               # max. longitudinal brake force [N]
    gamma_y_min = -np.inf                           # min. lateral wheel load transfer [N]
    gamma_y_max = np.inf                            # max. lateral wheel load transfer [N]

    # ------------------------------------------------------------------------------------------------------------------
    # STATE BOUNDARIES -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    v_min = 1.0 / v_s                               # min. velocity [m/s]
    v_max = pars["veh_params"]["v_max"] / v_s       # max. velocity [m/s]
    beta_min = -0.5 * np.pi / beta_s                # min. side slip angle [rad]
    beta_max = 0.5 * np.pi / beta_s                 # max. side slip angle [rad]
    omega_z_min = - 0.5 * np.pi / omega_z_s         # min. yaw rate [rad/s]
    omega_z_max = 0.5 * np.pi / omega_z_s           # max. yaw rate [rad/s]
    xi_min = - 0.5 * np.pi / xi_s                   # min. relative angle to tangent on reference line [rad]
    xi_max = 0.5 * np.pi / xi_s                     # max. relative angle to tangent on reference line [rad]

    # ------------------------------------------------------------------------------------------------------------------
    # INITIAL GUESS FOR DECISION VARIABLES -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    v_guess = 20.0 / v_s

    # ------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # continuous time dynamics
    f_dyn = ca.Function('f_dyn', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])

    # longitudinal tire forces [N]
    f_fx = ca.Function('f_fx', [x, u], [f_x_fl, f_x_fr, f_x_rl, f_x_rr],
                       ['x', 'u'], ['f_x_fl', 'f_x_fr', 'f_x_rl', 'f_x_rr'])
    # lateral tire forces [N]
    f_fy = ca.Function('f_fy', [x, u], [f_y_fl, f_y_fr, f_y_rl, f_y_rr],
                       ['x', 'u'], ['f_y_fl', 'f_y_fr', 'f_y_rl', 'f_y_rr'])
    # vertical tire forces [N]
    f_fz = ca.Function('f_fz', [x, u], [f_z_fl, f_z_fr, f_z_rl, f_z_rr],
                       ['x', 'u'], ['f_z_fl', 'f_z_fr', 'f_z_rl', 'f_z_rr'])

    # longitudinal and lateral acceleration [m/s²]
    f_a = ca.Function('f_a', [x, u], [ax, ay], ['x', 'u'], ['ax', 'ay'])

    if pars["pwr_params_mintime"]["pwr_behavior"]:

        machine.ini_nlp_state(x=x, u=u)
        inverter.ini_nlp_state(x=x, u=u)
        batt.ini_nlp_state(x=x, u=u)
        radiators.ini_nlp_state(x=x, u=u)

    # ------------------------------------------------------------------------------------------------------------------
    # FORMULATE NONLINEAR PROGRAM --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # initialize NLP vectors
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # initialize ouput vectors
    x_opt = []
    u_opt = []
    dt_opt = []
    tf_opt = []
    ax_opt = []
    ay_opt = []
    ec_opt = []

    # initialize control vectors (for regularization)
    delta_p = []
    F_p = []

    # boundary constraint: lift initial conditions
    Xk = ca.MX.sym('X0', nx)
    w.append(Xk)
    n_min = (-w_tr_right_interp(0) + pars["optim_opts"]["width_opt"] / 2) / n_s
    n_max = (w_tr_left_interp(0) - pars["optim_opts"]["width_opt"] / 2) / n_s
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min,
                    machine.temp_min, batt.temp_min, inverter.temp_min,
                    radiators.temp_cool_mi_min, radiators.temp_cool_b_min,
                    batt.soc_min])
        ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max,
                    machine.temp_max, batt.temp_max, inverter.temp_max,
                    radiators.temp_cool_mi_max, radiators.temp_cool_b_max,
                    batt.soc_max])
        w0.append([v_guess, 0.0, 0.0, 0.0, 0.0,
                   machine.temp_guess, batt.temp_guess, inverter.temp_guess,
                   radiators.temp_cool_mi_guess, radiators.temp_cool_b_guess,
                   batt.soc_guess])

        # Initial powertrain conditions
        g.append(Xk[5] - pwr_pars["T_mot_ini"] / machine.temp_mot_s)
        lbg.append([0])
        ubg.append([0])

        g.append(Xk[6] - pwr_pars["T_batt_ini"] / batt.temp_batt_s)
        lbg.append([0])
        ubg.append([0])

        g.append(Xk[7] - pwr_pars["T_inv_ini"] / inverter.temp_inv_s)
        lbg.append([0])
        ubg.append([0])

        g.append(Xk[8] - pwr_pars["T_cool_mi_ini"] / radiators.temp_cool_mi_s)
        lbg.append([0])
        ubg.append([0])

        g.append(Xk[9] - pwr_pars["T_cool_b_ini"] / radiators.temp_cool_b_s)
        lbg.append([0])
        ubg.append([0])

        g.append(Xk[10] - pwr_pars["SOC_ini"] / batt.soc_batt_s)
        lbg.append([0])
        ubg.append([0])

    else:
        lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
        ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max])
        w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])
    x_opt.append(Xk * x_s)

    # loop along the racetrack and formulate path constraints & system dynamic
    # retrieve step-sizes of optimization along reference line
    h = np.diff(s_opt)
    for k in range(N):
        # add decision variables for the control
        Uk = ca.MX.sym('U_' + str(k), nu)
        w.append(Uk)
        lbw.append([delta_min, f_drive_min, f_brake_min, gamma_y_min])
        ubw.append([delta_max, f_drive_max, f_brake_max, gamma_y_max])
        w0.append([0.0] * nu)

        # add decision variables for the state at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf] * nx)
            ubw.append([np.inf] * nx)
            if pars["pwr_params_mintime"]["pwr_behavior"]:
                w0.append([v_guess, 0.0, 0.0, 0.0, 0.0,
                           machine.temp_guess, batt.temp_guess, inverter.temp_guess,
                           radiators.temp_cool_mi_guess, radiators.temp_cool_b_guess,
                           batt.soc_guess])
            else:
                w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])

        # loop over all collocation points
        Xk_end = D[0] * Xk
        sf_opt = []
        for j in range(1, d + 1):
            # calculate the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]

            # interpolate kappa at the collocation point
            kappa_col = kappa_interp(k + tau[j])

            # append collocation equations (system dynamic)
            fj, qj = f_dyn(Xc[j - 1], Uk, kappa_col)
            g.append(h[k] * fj - xp)
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)

            # add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]

            # add contribution to quadrature function
            J = J + B[j] * qj * h[k]

            # add contribution to scaling factor (for calculating lap time)
            sf_opt.append(B[j] * qj * h[k])

        # calculate used energy 
        dt_opt.append(sf_opt[0] + sf_opt[1] + sf_opt[2])
        if pars["pwr_params_mintime"]["pwr_behavior"]:
            # Add battery output power [kW] and battery loss power [kW] to retireve entire system power [W] and
            # multiply by dt for energy consumption [Ws]
            ec_opt.append((batt.f_nlp(Xk, Uk)[0] + batt.f_nlp(Xk, Uk)[1]) * 1000 * dt_opt[-1])
        else:
            ec_opt.append(Xk[0] * v_s * Uk[1] * f_drive_s * dt_opt[-1])

        # add new decision variables for state at end of the collocation interval
        Xk = ca.MX.sym('X_' + str(k + 1), nx)
        w.append(Xk)
        n_min = (-w_tr_right_interp(k + 1) + pars["optim_opts"]["width_opt"] / 2.0) / n_s
        n_max = (w_tr_left_interp(k + 1) - pars["optim_opts"]["width_opt"] / 2.0) / n_s
        if pars["pwr_params_mintime"]["pwr_behavior"]:
            lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min,
                        machine.temp_min, batt.temp_min, inverter.temp_min,
                        radiators.temp_cool_mi_min, radiators.temp_cool_b_min,
                        batt.soc_min])
            ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max,
                        machine.temp_max, batt.temp_max, inverter.temp_max,
                        radiators.temp_cool_mi_max, radiators.temp_cool_b_max,
                        batt.soc_max])
            w0.append([v_guess, 0.0, 0.0, 0.0, 0.0,
                       machine.temp_guess, batt.temp_guess, inverter.temp_guess,
                       radiators.temp_cool_mi_guess, radiators.temp_cool_mi_guess,
                       batt.soc_guess])
        else:
            lbw.append([v_min, beta_min, omega_z_min, n_min, xi_min])
            ubw.append([v_max, beta_max, omega_z_max, n_max, xi_max])
            w0.append([v_guess, 0.0, 0.0, 0.0, 0.0])

        # add equality constraint
        g.append(Xk_end - Xk)
        lbg.append([0.0] * nx)
        ubg.append([0.0] * nx)

        # get tire forces
        f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
        f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)
        f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(Xk, Uk)

        # get accelerations (longitudinal + lateral)
        axk, ayk = f_a(Xk, Uk)

        # path constraint: limitied engine power
        g.append(Xk[0] * Uk[1])
        lbg.append([-np.inf])
        ubg.append([veh["power_max"] / (f_drive_s * v_s)])

        # get constant friction coefficient
        if pars["optim_opts"]["var_friction"] is None:
            mue_fl = pars["optim_opts"]["mue"]
            mue_fr = pars["optim_opts"]["mue"]
            mue_rl = pars["optim_opts"]["mue"]
            mue_rr = pars["optim_opts"]["mue"]

        # calculate variable friction coefficients along the reference line (regression with linear equations)
        elif pars["optim_opts"]["var_friction"] == "linear":
            # friction coefficient for each tire
            mue_fl = w_mue_fl[k + 1, 0] * Xk[3] * n_s + w_mue_fl[k + 1, 1]
            mue_fr = w_mue_fr[k + 1, 0] * Xk[3] * n_s + w_mue_fr[k + 1, 1]
            mue_rl = w_mue_rl[k + 1, 0] * Xk[3] * n_s + w_mue_rl[k + 1, 1]
            mue_rr = w_mue_rr[k + 1, 0] * Xk[3] * n_s + w_mue_rr[k + 1, 1]

        # calculate variable friction coefficients along the reference line (regression with gaussian basis functions)
        elif pars["optim_opts"]["var_friction"] == "gauss":
            # gaussian basis functions
            sigma = 2.0 * center_dist[k + 1, 0]
            n_gauss = pars["optim_opts"]["n_gauss"]
            n_q = np.linspace(-n_gauss, n_gauss, 2 * n_gauss + 1) * center_dist[k + 1, 0]

            gauss_basis = []
            for i in range(2 * n_gauss + 1):
                gauss_basis.append(ca.exp(-(Xk[3] * n_s - n_q[i]) ** 2 / (2 * (sigma ** 2))))
            gauss_basis = ca.vertcat(*gauss_basis)

            mue_fl = ca.dot(w_mue_fl[k + 1, :-1], gauss_basis) + w_mue_fl[k + 1, -1]
            mue_fr = ca.dot(w_mue_fr[k + 1, :-1], gauss_basis) + w_mue_fr[k + 1, -1]
            mue_rl = ca.dot(w_mue_rl[k + 1, :-1], gauss_basis) + w_mue_rl[k + 1, -1]
            mue_rr = ca.dot(w_mue_rr[k + 1, :-1], gauss_basis) + w_mue_rr[k + 1, -1]

        else:
            raise ValueError("No friction coefficients are available!")

        # path constraint: Kamm's Circle for each wheel
        g.append(((f_x_flk / (mue_fl * f_z_flk)) ** 2 + (f_y_flk / (mue_fl * f_z_flk)) ** 2))
        g.append(((f_x_frk / (mue_fr * f_z_frk)) ** 2 + (f_y_frk / (mue_fr * f_z_frk)) ** 2))
        g.append(((f_x_rlk / (mue_rl * f_z_rlk)) ** 2 + (f_y_rlk / (mue_rl * f_z_rlk)) ** 2))
        g.append(((f_x_rrk / (mue_rr * f_z_rrk)) ** 2 + (f_y_rrk / (mue_rr * f_z_rrk)) ** 2))
        lbg.append([0.0] * 4)
        ubg.append([1.0] * 4)

        # path constraint: lateral wheel load transfer
        g.append(((f_y_flk + f_y_frk) * ca.cos(Uk[0] * delta_s) + f_y_rlk + f_y_rrk
                  + (f_x_flk + f_x_frk) * ca.sin(Uk[0] * delta_s))
                 * veh["cog_z"] / ((veh["track_width_front"] + veh["track_width_rear"]) / 2) - Uk[3] * gamma_y_s)
        lbg.append([0.0])
        ubg.append([0.0])

        # path constraint: f_drive * f_brake == 0 (no simultaneous operation of brake and accelerator pedal)
        g.append(Uk[1] * Uk[2])
        lbg.append([-20000.0 / (f_drive_s * f_brake_s)])
        ubg.append([0.0])

        # path constraint: actor dynamic
        if k > 0:
            sigma = (1 - kappa_interp(k) * Xk[3] * n_s) / (Xk[0] * v_s)
            g.append((Uk - w[1 + (k - 1) * (nx - nx_pwr)]) / (h[k - 1] * sigma))
            lbg.append([delta_min / (veh["t_delta"]), -np.inf, f_brake_min / (veh["t_brake"]), -np.inf])
            ubg.append([delta_max / (veh["t_delta"]), f_drive_max / (veh["t_drive"]), np.inf, np.inf])

        # path constraint: safe trajectories with acceleration ellipse
        if pars["optim_opts"]["safe_traj"]:
            g.append((ca.fmax(axk, 0) / pars["optim_opts"]["ax_pos_safe"]) ** 2
                     + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
            g.append((ca.fmin(axk, 0) / pars["optim_opts"]["ax_neg_safe"]) ** 2
                     + (ayk / pars["optim_opts"]["ay_safe"]) ** 2)
            lbg.append([0.0] * 2)
            ubg.append([1.0] * 2)

        # append controls (for regularization)
        delta_p.append(Uk[0] * delta_s)
        F_p.append(Uk[1] * f_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

        # append outputs
        x_opt.append(Xk * x_s)
        u_opt.append(Uk * u_s)
        tf_opt.extend([f_x_flk, f_y_flk, f_z_flk, f_x_frk, f_y_frk, f_z_frk])
        tf_opt.extend([f_x_rlk, f_y_rlk, f_z_rlk, f_x_rrk, f_y_rrk, f_z_rrk])
        ax_opt.append(axk)
        ay_opt.append(ayk)

        if pars["pwr_params_mintime"]["pwr_behavior"]:
            machine.p_losses_opt.extend(machine.f_nlp(Xk, Uk))
            inverter.p_losses_opt.extend(inverter.f_nlp(Xk, Uk))
            batt.p_losses_opt.extend(batt.f_nlp(Xk, Uk))
            radiators.temps_opt.extend(radiators.f_nlp(Xk, Uk))

    # boundary constraint: start states = final states
    g.append(w[0] - Xk)
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        lbg.append([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        ubg.append([0.0, 0.0, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    else:
        lbg.append([0.0, 0.0, 0.0, 0.0, 0.0])
        ubg.append([0.0, 0.0, 0.0, 0.0, 0.0])

    # path constraint: limited energy consumption
    if pars["optim_opts"]["limit_energy"]:
        g.append(ca.sum1(ca.vertcat(*ec_opt)) / 3600000.0)
        lbg.append([0])
        ubg.append([pars["optim_opts"]["energy_limit"]])

    # formulate differentiation matrix (for regularization)
    diff_matrix = np.eye(N)
    for i in range(N - 1):
        diff_matrix[i, i + 1] = -1.0
    diff_matrix[N - 1, 0] = -1.0

    # regularization (delta)
    delta_p = ca.vertcat(*delta_p)
    Jp_delta = ca.mtimes(ca.MX(diff_matrix), delta_p)
    Jp_delta = ca.dot(Jp_delta, Jp_delta)

    # regularization (f_drive + f_brake)
    F_p = ca.vertcat(*F_p)
    Jp_f = ca.mtimes(ca.MX(diff_matrix), F_p)
    Jp_f = ca.dot(Jp_f, Jp_f)

    # formulate objective
    J = J + pars["optim_opts"]["penalty_F"] * Jp_f + pars["optim_opts"]["penalty_delta"] * Jp_delta

    # concatenate NLP vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # concatenate output vectors
    x_opt = ca.vertcat(*x_opt)
    u_opt = ca.vertcat(*u_opt)
    tf_opt = ca.vertcat(*tf_opt)
    dt_opt = ca.vertcat(*dt_opt)
    ax_opt = ca.vertcat(*ax_opt)
    ay_opt = ca.vertcat(*ay_opt)
    ec_opt = ca.vertcat(*ec_opt)
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        machine.p_losses_opt = ca.vertcat(*machine.p_losses_opt)
        inverter.p_losses_opt = ca.vertcat(*inverter.p_losses_opt)
        batt.p_losses_opt = ca.vertcat(*batt.p_losses_opt)
        radiators.temps_opt = ca.vertcat(*radiators.temps_opt)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE NLP SOLVER ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # fill nlp structure
    nlp = {'f': J, 'x': w, 'g': g}

    # solver options
    opts = {"expand": True,
            "verbose": print_debug,
            "ipopt.max_iter": 2000,
            "ipopt.tol": 1e-7}

    # solver options for warm start
    if pars["optim_opts"]["warm_start"]:
        opts_warm_start = {"ipopt.warm_start_init_point": "yes",
                           "ipopt.warm_start_bound_push": 1e-3,
                           "ipopt.warm_start_mult_bound_push": 1e-3,
                           "ipopt.warm_start_slack_bound_push": 1e-3,
                           "ipopt.mu_init": 1e-3}
        opts.update(opts_warm_start)

    # load warm start files
    if pars["optim_opts"]["warm_start"]:
        try:
            w0 = np.loadtxt(os.path.join(export_path, 'w0.csv'))
            lam_x0 = np.loadtxt(os.path.join(export_path, 'lam_x0.csv'))
            lam_g0 = np.loadtxt(os.path.join(export_path, 'lam_g0.csv'))
        except IOError:
            print('\033[91m' + 'WARNING: Failed to load warm start files!' + '\033[0m')
            sys.exit(1)

    # check warm start files
    if pars["optim_opts"]["warm_start"] and not len(w0) == len(lbw):
        print('\033[91m' + 'WARNING: Warm start files do not fit to the dimension of the NLP!' + '\033[0m')
        sys.exit(1)

    # create solver instance
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE NLP --------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # start time measure
    t0 = time.perf_counter()

    # solve NLP
    if pars["optim_opts"]["warm_start"]:
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, lam_x0=lam_x0, lam_g0=lam_g0)
    else:
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    # end time measure
    tend = time.perf_counter()

    if solver.stats()['return_status'] != 'Solve_Succeeded':
        print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
        sys.exit(1)

    # ------------------------------------------------------------------------------------------------------------------
    # EXTRACT SOLUTION -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # helper function to extract solution for state variables, control variables, tire forces, time
    f_sol = ca.Function('f_sol', [w], [x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt],
                        ['w'], ['x_opt', 'u_opt', 'tf_opt', 'dt_opt', 'ax_opt', 'ay_opt', 'ec_opt'])

    if pars["pwr_params_mintime"]["pwr_behavior"]:

        machine.extract_sol(w=w,
                            sol_states=sol['x'])
        inverter.extract_sol(w=w,
                             sol_states=sol['x'])
        batt.extract_sol(w=w,
                         sol_states=sol['x'])
        radiators.extract_sol(w=w,
                              sol_states=sol['x'])

        # Store for convenient export
        pwr_comps = {"machine": machine,
                     "inverter": inverter,
                     "batt": batt,
                     "radiators": radiators}
    else:

        pwr_comps = None

    # extract solution
    x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt = f_sol(sol['x'])

    # solution for state variables
    x_opt = np.reshape(x_opt, (N + 1, nx))

    # solution for control variables
    u_opt = np.reshape(u_opt, (N, nu))

    # solution for tire forces
    tf_opt = np.append(tf_opt[-12:], tf_opt[:])
    tf_opt = np.reshape(tf_opt, (N + 1, 12))

    # solution for time
    t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

    # solution for acceleration
    ax_opt = np.append(ax_opt[-1], ax_opt)
    ay_opt = np.append(ay_opt[-1], ay_opt)
    atot_opt = np.sqrt(np.power(ax_opt, 2) + np.power(ay_opt, 2))

    # solution for energy consumption
    ec_opt_cum = np.hstack((0.0, np.cumsum(ec_opt))) / 3600.0

    # ------------------------------------------------------------------------------------------------------------------
    # EXPORT SOLUTION --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # export data to CSVs
    opt_mintime_traj.src.export_mintime_solution.export_mintime_solution(file_path=export_path,
                                                                         pars=pars,
                                                                         s=s_opt,
                                                                         t=t_opt,
                                                                         x=x_opt,
                                                                         u=u_opt,
                                                                         tf=tf_opt,
                                                                         ax=ax_opt,
                                                                         ay=ay_opt,
                                                                         atot=atot_opt,
                                                                         w0=sol["x"],
                                                                         lam_x0=sol["lam_x"],
                                                                         lam_g0=sol["lam_g"],
                                                                         pwr=pwr_comps)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT & PRINT RESULTS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_debug:
        opt_mintime_traj.src.result_plots_mintime.result_plots_mintime(pars=pars,
                                                                       reftrack=reftrack,
                                                                       s=s_opt,
                                                                       t=t_opt,
                                                                       x=x_opt,
                                                                       u=u_opt,
                                                                       ax=ax_opt,
                                                                       ay=ay_opt,
                                                                       atot=atot_opt,
                                                                       tf=tf_opt,
                                                                       ec=ec_opt_cum,
                                                                       pwr=pwr_comps)

    if print_debug:
        print("INFO: Laptime: %.3fs" % t_opt[-1])
        print("INFO: NLP solving time: %.3fs" % (tend - t0))
        print("INFO: Maximum abs(ay): %.2fm/s2" % np.amax(ay_opt))
        print("INFO: Maximum ax: %.2fm/s2" % np.amax(ax_opt))
        print("INFO: Minimum ax: %.2fm/s2" % np.amin(ax_opt))
        print("INFO: Maximum total acc: %.2fm/s2" % np.amax(atot_opt))
        print('INFO: Energy consumption: %.3fWh' % ec_opt_cum[-1])

    return -x_opt[:-1, 3], x_opt[:-1, 0], reftrack, a_interp, normvectors


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
