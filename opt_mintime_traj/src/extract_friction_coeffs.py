import numpy as np
import matplotlib.pyplot as plt
import math
import trajectory_planning_helpers as tph
import opt_mintime_traj


def extract_friction_coeffs(reftrack: np.ndarray,
                            normvectors: np.ndarray,
                            tpamap_path: str,
                            tpadata_path: str,
                            pars: dict,
                            dn: float,
                            print_debug: bool,
                            plot_debug: bool) -> tuple:
    """
    Created by:
    Fabian Christ

    Documentation:
    Extracting friction coefficients on a fine grid on the normal vectors along the racetrack from the provided
    friction map.

    Inputs:
    reftrack:       track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvectors:    array containing normalized normal vectors for every traj. point [x_component, y_component]
    tpamap_path:    file path to tpa map (required for friction map loading)
    tpadata_path:   file path to tpa data (required for friction map loading)
    pars:           parameters dictionary
    dn:             distance of equidistant points on normal vectors for extracting the friction coefficients
    print_debug:    determines if debug prints are shown
    plot_debug:     determines if debug plots are shown

    Outputs:
    n:              lateral distance of equidistant points on normal vectors along the racetrack
    mue_fl:         grid of friction coefficients along the racetrack (left front wheel)
    mue_fr:         grid of friction coefficients along the racetrack (right front wheel)
    mue_rl:         grid of friction coefficients along the racetrack (left rear wheel)
    mue_rr:         grid of friction coefficients along the racetrack (right rear wheel)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATION ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # track data
    reftrack_cl = np.vstack([reftrack, reftrack[0]])
    refline_cl = reftrack_cl[:, :2]
    track_width_right_cl = reftrack_cl[:, 2]
    track_width_left_cl = reftrack_cl[:, 3]
    normvectors_cl = np.vstack([normvectors, normvectors[0]])
    tang_vec = np.asarray([-normvectors_cl[:, 1], normvectors_cl[:, 0]]).T

    # number of steps along the reference line
    num_steps = len(reftrack_cl[:, 0])

    # vehicle data
    width = pars["optim_opts"]["width_opt"]
    wb_f = pars["vehicle_params_mintime"]["wheelbase_front"]
    wb_r = pars["vehicle_params_mintime"]["wheelbase_rear"]

    # initialize map interface
    map_interface = opt_mintime_traj.src.friction_map_interface.FrictionMapInterface(tpamap_path=tpamap_path,
                                                                                     tpadata_path=tpadata_path)

    # initialize solution
    n = []
    mue_fl = []
    mue_fr = []
    mue_rl = []
    mue_rr = []

    # plot position of extracted friction coefficients
    if plot_debug:
        plt.figure(0) 

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATION ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    for i in range(num_steps):
        # number of required points on the normal vector to the left and right side
        num_right = math.floor((track_width_right_cl[i] - 0.5 * width - 0.5) / dn)
        num_left = math.floor((track_width_left_cl[i] - 0.5 * width - 0.5) / dn)

        # vector of lateral distances on the normal vector for the vehicle's center of gravity
        n_pos = np.linspace(-dn * num_right, dn * num_left, num_right + num_left + 1)
        n.append(n_pos)

        # initialize xy coordinates for each point on the normal and each tire
        xy = np.zeros((8, num_right + num_left + 1))

        # calculate xy coordinates for each point on the normal and each tire
        for j in range(num_right + num_left + 1):
            xy[0:2, j] = (refline_cl[i, :] + wb_f * tang_vec[i, :]) - (n_pos[j] + 0.5 * width) * normvectors_cl[i, :]
            xy[2:4, j] = (refline_cl[i, :] + wb_f * tang_vec[i, :]) - (n_pos[j] - 0.5 * width) * normvectors_cl[i, :]
            xy[4:6, j] = (refline_cl[i, :] - wb_r * tang_vec[i, :]) - (n_pos[j] + 0.5 * width) * normvectors_cl[i, :]
            xy[6:8, j] = (refline_cl[i, :] - wb_r * tang_vec[i, :]) - (n_pos[j] - 0.5 * width) * normvectors_cl[i, :]

        # get friction coefficients for these coordinates
        mue_fl.append(map_interface.get_friction_singlepos(xy[0:2, :].T))
        mue_fr.append(map_interface.get_friction_singlepos(xy[2:4, :].T))
        mue_rl.append(map_interface.get_friction_singlepos(xy[4:6, :].T))
        mue_rr.append(map_interface.get_friction_singlepos(xy[6:8, :].T))

        if print_debug:
            tph.progressbar.progressbar(i, num_steps, 'Extraction of Friction Coefficients from Friction Map')

        if plot_debug:
            plt.plot(xy[0, :], xy[1, :], '.')
            plt.plot(xy[2, :], xy[3, :], '.')
            plt.plot(xy[4, :], xy[5, :], '.')
            plt.plot(xy[6, :], xy[7, :], '.')

    if plot_debug:
        bound_r = reftrack[:, :2] + normvectors * np.expand_dims(reftrack[:, 2], 1)
        bound_l = reftrack[:, :2] - normvectors * np.expand_dims(reftrack[:, 3], 1)
        plt.plot(reftrack[:, 0], reftrack[:, 1], color='grey')
        plt.plot(bound_r[:, 0], bound_r[:, 1], color='black')
        plt.plot(bound_l[:, 0], bound_l[:, 1], color='black')
        plt.title('Extraction of Friction Coefficients from Friction Map')
        plt.grid()
        plt.axis('equal')
        plt.show()

    return n, mue_fl, mue_fr, mue_rl, mue_rr


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
