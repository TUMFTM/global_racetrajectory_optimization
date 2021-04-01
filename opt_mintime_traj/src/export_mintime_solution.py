import numpy as np
import os


def export_mintime_solution(file_path: str,
                            pars: dict,
                            s: np.ndarray,
                            t: np.ndarray,
                            x: np.ndarray,
                            u: np.ndarray,
                            tf: np.ndarray,
                            ax: np.ndarray,
                            ay: np.ndarray,
                            atot: np.ndarray,
                            w0: np.ndarray,
                            lam_x0: np.ndarray,
                            lam_g0: np.ndarray,
                            pwr: dict = None) -> None:

    """
    Created by:
    Fabian Christ

    Modified by:
    Thomas Herrmann (thomas.herrmann@tum.de)

    Documentation:
    This function is used to export the solution of the time-optimal trajectory planning into several csv files.

    Inputs:
    file_path:      path for the output files
    t:              solution for the time along the reference line (at corresponding station s)
    s:              station along the reference line (at corresponding time t)
    x:              solution for the state variables (at corresponding time t / station s)
    u:              solution for the control variables (at corresponding time t / station s)
    tf:             solution for the tire forces (at corresponding time t / station s)
    ax:             solution for the longitudinal acceleration (at corresponding time t / station s)
    ay:             solution for the lateral acceleration (at corresponding time t / station s)
    atot:           solution for the total acceleration (at corresponding time t / station s)
    w0:             solution for all decision variables (for warm starting the nonlinear program)
    lam_x0:         solution for the lagrange multipliers (for warm starting the nonlinear program)
    lam_g0:         solution for the lagrange multipliers (for warm starting the nonlinear program)
    """

    # save state variables
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        header_x = ("s_m; t_s; v_mps; beta_rad; omega_z_radps; n_m; xi_rad; "
                    "machine.temp_mot_dC; batt.temp_batt_dC; inverter.temp_inv_dC; "
                    "radiators.temp_cool_mi_dC; radiators.temp_cool_b_dC; batt.soc_batt")
        fmt_x = "%.1f; %.3f; %.2f; %.5f; %.5f; %.5f; %.5f; %.2f; %.2f; %.2f; %.2f; %.2f; %.5f;"
        states = np.column_stack((s, t, x))
        np.savetxt(os.path.join(file_path, 'states.csv'), states, fmt=fmt_x, header=header_x)
    else:
        header_x = "s_m; t_s; v_mps; beta_rad; omega_z_radps; n_m; xi_rad"
        fmt_x = "%.1f; %.3f; %.2f; %.5f; %.5f; %.5f; %.5f"
        states = np.column_stack((s, t, x))
        np.savetxt(os.path.join(file_path, 'states.csv'), states, fmt=fmt_x, header=header_x)

    # save control variables
    header_u = "s_m; t_s; delta_rad; f_drive_N; f_brake_N; gamma_y_N"
    fmt_u = "%.1f; %.3f; %.5f; %.1f; %.1f; %.1f"
    controls = np.column_stack((s[:-1], t[:-1], u))
    np.savetxt(os.path.join(file_path, 'controls.csv'), controls, fmt=fmt_u, header=header_u)

    # save tire forces
    header_tf = "s_m; t_s; f_x_fl_N; f_y_fl_N; f_z_fl_N; f_x_fr_N; f_y_fr_N; f_z_fr_N;" \
                "f_x_rl_N; f_y_rl_N; f_z_rl_N; f_x_rr_N;f_y_rr_N; f_z_rr_N;"
    fmt_tf = "%.1f; %.3f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f; %.1f"
    tire_forces = np.column_stack((s, t, tf))
    np.savetxt(os.path.join(file_path, 'tire_forces.csv'), tire_forces, fmt=fmt_tf, header=header_tf)

    # save accelerations
    header_a = "s_m; t_s; ax_mps2; ay_mps2; atot_mps2"
    fmt_a = "%.1f; %.3f; %.3f; %.3f; %.3f"
    accelerations = np.column_stack((s, t, ax, ay, atot))
    np.savetxt(os.path.join(file_path, 'accelerations.csv'), accelerations, fmt=fmt_a, header=header_a)

    # save power losses
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        if pars["pwr_params_mintime"]["simple_loss"]:
            header_pwr_l = \
                ("s_m; t_s; "
                 "P_loss_1machine_kW; "
                 "P_loss_1inverter_kW; "
                 "P_loss_batt_kW; P_out_batt_kW")
            fmt_pwr_l = ("%.1f; %.3f; "
                         "%.2f; "
                         "%.2f; "
                         "%.2f; %.2f")
            pwr_losses = \
                np.column_stack((s[:-1], t[:-1],
                                 pwr["machine"].p_loss_total,
                                 pwr["inverter"].p_loss_total,
                                 pwr["batt"].p_loss_total, pwr["batt"].p_out_batt))
        else:
            header_pwr_l = \
                ("s_m; t_s; "
                 "P_loss_1machine_kW; P_loss_copper_1machine_kW; "
                 "P_loss_statorIron_1machine_kW; P_loss_rotor_1machine_kW; "
                 "P_loss_1inverter_kW; P_loss_switch_1inverter_kW; P_loss_cond_1inverter; "
                 "P_loss_batt_kW; P_out_batt_kW")
            fmt_pwr_l = ("%.1f; %.3f; "
                         "%.2f; %.2f; "
                         "%.2f; %.2f; "
                         "%.2f; %.2f; %.2f; "
                         "%.2f; %.2f")
            pwr_losses = \
                np.column_stack((s[:-1], t[:-1],
                                 pwr["machine"].p_loss_total, pwr["machine"].p_loss_copper,
                                 pwr["machine"].p_loss_stator_iron, pwr["machine"].p_loss_rotor,
                                 pwr["inverter"].p_loss_total, pwr["inverter"].p_loss_switch, pwr["inverter"].p_loss_cond,
                                 pwr["batt"].p_loss_total, pwr["batt"].p_out_batt))

        np.savetxt(os.path.join(file_path, 'power_losses.csv'), pwr_losses, fmt=fmt_pwr_l, header=header_pwr_l)

    # save solution of decision variables and lagrange multipliers
    np.savetxt(os.path.join(file_path, 'w0.csv'), w0, delimiter=';')
    np.savetxt(os.path.join(file_path, 'lam_x0.csv'), lam_x0, delimiter=';')
    np.savetxt(os.path.join(file_path, 'lam_g0.csv'), lam_g0, delimiter=';')


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
