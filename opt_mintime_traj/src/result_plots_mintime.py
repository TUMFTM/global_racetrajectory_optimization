import numpy as np
import matplotlib.pyplot as plt


def result_plots_mintime(pars: dict,
                         reftrack: np.ndarray,
                         s: np.ndarray,
                         t: np.ndarray,
                         x: np.ndarray,
                         u: np.ndarray,
                         ax: np.ndarray,
                         ay: np.ndarray,
                         atot: np.ndarray,
                         tf: np.ndarray,
                         ec: np.ndarray,
                         pwr: dict = None) -> None:

    """
    Created by:
    Fabian Christ

    Extended by:
    Thomas Herrmann (thomas.herrmann@tum.de)

    Documentation:
    This function plots several figures containing relevant trajectory information after trajectory optimization.

    Inputs:
    pars:       parameters dictionary
    reftrack:   contains the information of the reftrack -> [x, y, w_tr_right, w_tr_left]
    s:          contains the curvi-linear distance along the trajectory
    t:          contains the time along the trajectory
    x:          contains all state variables along the trajectory
    u:          contains all control variables along the trajectory
    ax:         contains the longitudinal acceleration along the trajectory
    ay:         contains the lateral acceleration along the trajectory
    atot:       contains the total acceleration along the trajectory
    tf:         contains all tire forces along the trajectory
    ec:         contains the used energy along the trajectory
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT OPTIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    plt.rcParams['axes.labelsize'] = 10.0
    plt.rcParams['axes.titlesize'] = 11.0
    plt.rcParams['legend.fontsize'] = 10.0
    plt.rcParams['figure.figsize'] = 25 / 2.54, 20 / 2.54

    plot_opts = {"v_a_t": True,
                 "general": True,
                 "lateral_distance": True,
                 "power": True,
                 "kamm_circle": True,
                 "tire_forces": True,
                 "tire_forces_longitudinal": True,
                 "tire_forces_dynamic": True,
                 "energy_consumption": True,
                 "pwr_states": True,
                 "pwr_soc": True,
                 "pwr_losses": True}

    # ------------------------------------------------------------------------------------------------------------------
    # EXTRACT PLOT DATA ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    # state variables
    v = x[:, 0]
    beta = x[:, 1]
    omega_z = x[:, 2]
    n = x[:, 3]
    xi = x[:, 4]
    if pars["pwr_params_mintime"]["pwr_behavior"]:
        temp_mot = x[:, 5]
        temp_batt = x[:, 6]
        temp_inv = x[:, 7]
        temp_radiators_cool_mi = x[:, 8]
        temp_radiators_cool_b = x[:, 9]
        soc_batt = x[:, 10]

    # control variables
    delta = np.append(u[:, 0], u[0, 0])
    f_drive = np.append(u[:, 1], u[0, 1])
    f_brake = np.append(u[:, 2], u[0, 2])
    gamma_y = np.append(u[:, 3], u[0, 3])
    
    # tire forces
    tf_x_fl = tf[:, 0]
    tf_y_fl = tf[:, 1]
    tf_z_fl = tf[:, 2]
    tf_x_fr = tf[:, 3]
    tf_y_fr = tf[:, 4]
    tf_z_fr = tf[:, 5] 
    tf_x_rl = tf[:, 6]
    tf_y_rl = tf[:, 7]
    tf_z_rl = tf[:, 8]
    tf_x_rr = tf[:, 9]
    tf_y_rr = tf[:, 10]
    tf_z_rr = tf[:, 11]

    # parameters
    g = pars["veh_params"]["g"]
    veh = pars["vehicle_params_mintime"]
    tire = pars["tire_params_mintime"]

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: VELOCITY + LONGITUDINAL ACCELERATION + LATERAL ACCELERATION + TOTAL ACCELERATION + TIME --------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["v_a_t"]:

        plt.figure(1)
        plt.clf()
        plt.plot(s, v)
        plt.plot(s, ax)
        plt.plot(s, ay)
        plt.plot(s, atot)
        plt.plot(s, t)

        plt.grid()
        plt.ylim(bottom=-15)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.legend([r'$\it{v}$' + ' in ' + r'$\it{\frac{m}{s}}$',
                    r'$\it{a_x}$' + ' in ' + r'$\it{\frac{m}{s^2}}$',
                    r'$\it{a_y}$' + ' in ' + r'$\it{\frac{m}{s^2}}$',
                    r'$\it{a_{tot}}$' + ' in ' + r'$\it{\frac{m}{s^2}}$',
                    r'$\it{t}$' + ' in ' + r'$\it{s}$'])
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: SIDE SLIP ANGLE + YAW RATE + RELATIVE ANGLE TO TANGENT ON REFLINE + STEERING ANGLE -------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["general"]:

        plt.figure(2)
        plt.clf()
        plt.subplot(221)
        plt.plot(s, beta * 180 / np.pi)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('side slip angle ' + r'$\beta$' + ' in ' + r'$\it{°}$')
        plt.grid()
        plt.subplot(222)
        plt.plot(s, omega_z * 180 / np.pi)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('yaw rate ' + r'$\omega_{z}$' + ' in ' + r'$\it{\frac{°}{s}}$')
        plt.grid()
        plt.subplot(223)
        plt.plot(s, xi * 180 / np.pi)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('relative angle to tangent on reference line ' + r'$\xi$' + ' in ' + r'$\it{°}$')
        plt.grid()
        plt.subplot(224)
        plt.step(s, delta * 180 / np.pi, where='post')
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('steering angle ' + r'$\delta$' + ' in ' + r'$\it{°}$')
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: LATERAL DISTANCE TO REFERENCE LINE + ROAD BOUNDARIES -------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["lateral_distance"]:

        plt.figure(3)
        plt.clf()
        plt.plot(s, n)
        reftrack_cl = np.vstack((reftrack, reftrack[0, :]))
        plt.plot(s, reftrack_cl[:, 3], color='black')
        plt.plot(s, reftrack_cl[:, 3] - pars["optim_opts"]["width_opt"] / 2, color='grey')
        plt.plot(s, -reftrack_cl[:, 2], color='black')
        plt.plot(s, -reftrack_cl[:, 2] + pars["optim_opts"]["width_opt"] / 2, color='grey')
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('lateral distance to reference line ' + r'$\it{n}$' + ' in ' + r'$\it{m}$')
        plt.legend(['raceline', 'road boundaries', 'road boundaries - safety margin'], ncol=1, loc=4)
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: KAMM's CIRCLE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["kamm_circle"]:

        plt.figure(5)
        plt.clf()
        plt.suptitle("Kamm's Circle")
        plt.subplot(221)
        circle1 = plt.Circle((0, 0), 1, fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle1)
        plt.plot(tf_y_fl / (tf_z_fl * pars["optim_opts"]["mue"]),
                 tf_x_fl / (tf_z_fl * pars["optim_opts"]["mue"]), '^:')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel(r'$\it{\frac{F_{y}}{F_{ymax}}}$')
        plt.ylabel(r'$\it{\frac{F_{x}}{F_{xmax}}}$')
        plt.axis('equal')
        plt.grid()

        plt.subplot(222)
        circle1 = plt.Circle((0, 0), 1, fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle1)
        plt.plot(tf_y_fr / (tf_z_fr * pars["optim_opts"]["mue"]),
                 tf_x_fr / (tf_z_fr * pars["optim_opts"]["mue"]), '^:')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel(r'$\it{\frac{F_{y}}{F_{ymax}}}$')
        plt.ylabel(r'$\it{\frac{F_{x}}{F_{xmax}}}$')
        plt.axis('equal')
        plt.grid()

        plt.subplot(223)
        circle1 = plt.Circle((0, 0), 1, fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle1)
        plt.plot(tf_y_rl / (tf_z_rl * pars["optim_opts"]["mue"]),
                 tf_x_rl / (tf_z_rl * pars["optim_opts"]["mue"]), '^:')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel(r'$\it{\frac{F_{y}}{F_{ymax}}}$')
        plt.ylabel(r'$\it{\frac{F_{x}}{F_{xmax}}}$')
        plt.axis('equal')
        plt.grid()

        plt.subplot(224)
        circle1 = plt.Circle((0, 0), 1, fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle1)
        plt.plot(tf_y_rr / (tf_z_rr * pars["optim_opts"]["mue"]),
                 tf_x_rr / (tf_z_rr * pars["optim_opts"]["mue"]), '^:')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel(r'$\it{\frac{F_{y}}{F_{ymax}}}$')
        plt.ylabel(r'$\it{\frac{F_{x}}{F_{xmax}}}$')
        plt.axis('equal')
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: TIRE FORCES (LONGITUDINAL + LATERAL + NORMAL) --------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["tire_forces"]:

        plt.figure(6)
        plt.clf()
        plt.suptitle("Tire Forces")
        plt.subplot(221)
        plt.plot(s, tf_x_fl)
        plt.plot(s, tf_y_fl)
        plt.plot(s, tf_z_fl)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')
        plt.legend([r'$\it{F_{x}}$', r'$\it{F_{y}}$', r'$\it{F_{z}}$'], ncol=3, loc=4)
        plt.grid()

        plt.subplot(222)
        plt.plot(s, tf_x_fr)
        plt.plot(s, tf_y_fr)
        plt.plot(s, tf_z_fr)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')
        plt.legend([r'$\it{F_{x}}$', r'$\it{F_{y}}$', r'$\it{F_{z}}$'], ncol=3, loc=4)
        plt.grid()

        plt.subplot(223)
        plt.plot(s, tf_x_rl)
        plt.plot(s, tf_y_rl)
        plt.plot(s, tf_z_rl)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')
        plt.legend([r'$\it{F_{x}}$', r'$\it{F_{y}}$', r'$\it{F_{z}}$'], ncol=3, loc=4)
        plt.grid()

        plt.subplot(224)
        plt.plot(s, tf_x_rr)
        plt.plot(s, tf_y_rr)
        plt.plot(s, tf_z_rr)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')
        plt.legend([r'$\it{F_{x}}$', r'$\it{F_{y}}$', r'$\it{F_{z}}$'], ncol=3, loc=4)
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: TIRE FORCES (LONGITUDINAL) ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["tire_forces_longitudinal"]:
        plt.figure(7)
        plt.step(s, f_drive / 1000, where="post")
        plt.step(s, f_brake / 1000, where='post')
        plt.step(s, (f_drive + f_brake) / 1000, where='post')
        plt.plot(s, veh["power_max"] / (v * 1000), linewidth=0.5)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F}$' + ' in ' + r'$\it{kN}$')
        plt.legend([r'$\it{F_{drive}}$', r'$\it{F_{brake}}$',
                    r'$\it{F_{drive}}$' + " + " + r'$\it{F_{brake}}$',
                    r'$\it{F_{P_{max}}}$'], ncol=1, loc=4)
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: DYNAMIC WHEEL LOAD TRANSFER --------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["tire_forces_dynamic"]:

        f_xroll = tire["c_roll"] * pars["veh_params"]["mass"] * g
        f_xdrag = pars["veh_params"]["dragcoeff"] * v ** 2

        f_zlift_fl = 0.5 * veh["liftcoeff_front"] * v ** 2
        f_zlift_fr = 0.5 * veh["liftcoeff_front"] * v ** 2
        f_zlift_rl = 0.5 * veh["liftcoeff_rear"] * v ** 2
        f_zlift_rr = 0.5 * veh["liftcoeff_rear"] * v ** 2

        f_zlong_fl = -0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xroll - f_xdrag)
        f_zlong_fr = -0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_brake - f_xroll - f_xdrag)
        f_zlong_rl = 0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_drive - f_xroll - f_xdrag)
        f_zlong_rr = 0.5 * veh["cog_z"] / veh["wheelbase"] * (f_drive + f_drive - f_xroll - f_xdrag)

        f_zlat_fl = - veh["k_roll"] * gamma_y
        f_zlat_fr = veh["k_roll"] * gamma_y
        f_zlat_rl = - (1 - veh["k_roll"]) * gamma_y
        f_zlat_rr = (1 - veh["k_roll"]) * gamma_y

        plt.figure(8)
        plt.suptitle("Dynamic Wheel Load")
        plt.subplot(221)
        plt.plot(s, f_zlift_fl)
        plt.plot(s, f_zlong_fl)
        plt.plot(s, f_zlat_fl)
        plt.plot(s, f_zlift_fl + f_zlong_fl + f_zlat_fl, color='black')
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')

        plt.grid()
        plt.subplot(222)
        plt.plot(s, f_zlift_fr)
        plt.plot(s, f_zlong_fr)
        plt.plot(s, f_zlat_fr)
        plt.plot(s, f_zlift_fr + f_zlong_fr + f_zlat_fr, color='black')
        plt.xlabel(r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')

        plt.grid()
        plt.subplot(223)
        plt.plot(s, f_zlift_rl)
        plt.plot(s, f_zlong_rl)
        plt.plot(s, f_zlat_rl)
        plt.plot(s, f_zlift_rl + f_zlong_rl + f_zlat_rl, color='black')
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')

        plt.grid()
        plt.subplot(224)
        plt.plot(s, f_zlift_rr)
        plt.plot(s, f_zlong_rr)
        plt.plot(s, f_zlat_rr)
        plt.plot(s, f_zlift_rr + f_zlong_rr + f_zlat_rr, color='black')
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel(r'$\it{F_{i}}$' + ' in ' + r'$\it{N}$')
        plt.legend([r'$\it{F_{lift}}$', r'$\it{F_{dyn,long}}$', r'$\it{F_{dyn,lat}}$',
                    r'$\it{F_{lift}}$' + ' + ' + r'$\it{F_{dyn,long}}$' + ' + ' + r'$\it{F_{dyn,lat}}$'], ncol=2, loc=4)
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: ENERGY CONSUMPTION -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["energy_consumption"]:

        plt.figure(9)
        plt.clf()
        plt.plot(s, ec)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('energy consumption ' + r'$\it{ec}$' + ' in ' + r'$\it{Wh}$')
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: POWER ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["power"]:
        plt.figure(4)
        plt.clf()
        plt.plot(s, v * (f_drive + f_brake) / 1000.0)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$m$')
        plt.ylabel('power ' + r'$\it{P}$' + ' in ' + r'$kW$')
        plt.grid()
        plt.legend(r'$\it{P_{wheel}}$')
        if pwr is not None:
            plt.plot(s[:-1], pwr["batt"].p_loss_total + pwr["batt"].p_out_batt)
            plt.legend([r'$\it{P_{wheel}}$', r'$\it{P_{system}}$'])
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: POWERTRAIN TEMPERATURES ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if pars["pwr_params_mintime"]["pwr_behavior"] and plot_opts["pwr_states"]:

        plt.figure(10)
        plt.plot(s, temp_mot)
        plt.plot(s, temp_batt)
        plt.plot(s, temp_inv)
        plt.plot(s, temp_radiators_cool_mi)
        plt.plot(s, temp_radiators_cool_b)

        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('component temperatures ' + r'$\it{T}$' + ' in ' + r'°C')
        plt.legend([r'$\it{T_\mathrm{Machine}}$', r'$\it{T_\mathrm{Battery}}$', r'$\it{T_\mathrm{Inverter}}$',
                    r'$\it{T_\mathrm{Fluid_{MI}}}$', r'$\it{T_\mathrm{Fluid_B}}$'])
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: SOC BATTERY ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if pars["pwr_params_mintime"]["pwr_behavior"] and plot_opts["pwr_soc"]:

        plt.figure(11)
        plt.plot(s, soc_batt)
        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.ylabel('SOC battery [1 - 0]')
        plt.grid()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT: POWER LOSSES -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if pars["pwr_params_mintime"]["pwr_behavior"] and plot_opts["pwr_losses"]:

        if pars["pwr_params_mintime"]["simple_loss"]:
            plt.figure(12)
            plt.plot(s[:-1], pwr["machine"].p_loss_total)
            plt.plot(s[:-1], pwr["inverter"].p_loss_total)
            plt.plot(s[:-1], pwr["batt"].p_loss_total)
            plt.legend([r'$\it{P_\mathrm{loss,machine}}$', r'$\it{P_\mathrm{loss,inverter}}$',
                        r'$\it{P_\mathrm{loss,battery}}$'])
            plt.ylabel('Power loss ' + r'$\it{P_\mathrm{loss}}$' + ' in ' + r'kW')
        else:
            plt.figure(12)
            plt.subplot(311)
            plt.plot(s[:-1], pwr["machine"].p_loss_total)
            plt.plot(s[:-1], pwr["machine"].p_loss_copper)
            plt.plot(s[:-1], pwr["machine"].p_loss_stator_iron)
            plt.plot(s[:-1], pwr["machine"].p_loss_rotor)
            plt.ylabel('Power loss single machine\n' + r'$\it{P_\mathrm{loss}}$' + ' in ' + r'kW')
            plt.legend([r'$\it{P_\mathrm{loss,total}}$', r'$\it{P_\mathrm{loss,copper}}$',
                        r'$\it{P_\mathrm{loss,statorIron}}$', r'$\it{P_\mathrm{loss,rotor}}$'])
            plt.grid()
            plt.subplot(312)
            plt.plot(s[:-1], pwr["inverter"].p_loss_total)
            plt.plot(s[:-1], pwr["inverter"].p_loss_switch)
            plt.plot(s[:-1], pwr["inverter"].p_loss_cond)
            plt.legend([r'$\it{P_\mathrm{loss,total}}$', r'$\it{P_\mathrm{loss,switching}}$',
                        r'$\it{P_\mathrm{loss,conducting}}$'])
            plt.ylabel('Power loss single inverter\n' + r'$\it{P_\mathrm{loss}}$' + ' in ' + r'kW')
            plt.grid()
            plt.subplot(313)
            plt.plot(s[:-1], pwr["batt"].p_loss_total)
            plt.ylabel('Power loss battery\n' + r'$\it{P_\mathrm{loss}}$' + ' in ' + r'kW')

        plt.xlabel('distance ' + r'$\it{s}$' + ' in ' + r'$\it{m}$')
        plt.grid()
        plt.show()

# testing --------------------------------------------------------------------------------------------------------------
    if __name__ == "__main__":
        pass
