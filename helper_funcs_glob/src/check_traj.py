import numpy as np
import helper_funcs_glob


def check_traj(reftrack: np.ndarray,
               reftrack_normvec_normalized: np.ndarray,
               trajectory: np.ndarray,
               ggv: np.ndarray,
               veh_dims: dict,
               debug: bool) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function checks the generated trajectory in regards of minimum distance to the boundaries and maximum
    curvature and accelerations.

    Inputs:
    reftrack:       track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reftrack_normvec_normalized: normalized normal vectors on the reference line [x_m, y_m]
    trajectory:     trajectory to be checked [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    ggv:            ggv diagram [v_mps, ax_max_machines_mps2, ax_max_tires_mps2, ax_min_tires_mps2, ay_max_tires_mps2]
    veh_dims:       vehicle dimensions in m {l_veh_real, w_veh_real}
    debug:          boolean showing if debug messages should be printed

    Outputs:
    bound_r:        right track boundary [x_m, y_m]
    bound_l:        left track boundary [x_m, y_m]
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK VEHICLE EDGES FOR MINIMUM DISTANCE TO TRACK BOUNDARIES -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate boundaries and interpolate them to small stepsizes (currently linear interpolation)
    bound_r = reftrack[:, :2] + reftrack_normvec_normalized * np.expand_dims(reftrack[:, 2], 1)
    bound_l = reftrack[:, :2] - reftrack_normvec_normalized * np.expand_dims(reftrack[:, 3], 1)

    # check boundaries for vehicle edges
    bound_r_tmp = np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2))))
    bound_l_tmp = np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2))))

    bound_r_check = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound_r_tmp,
                                                                    stepsize_approx=1.0)[0]
    bound_l_check = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound_l_tmp,
                                                                    stepsize_approx=1.0)[0]

    # calculate minimum distances of every trajectory point to the boundaries
    min_dists = helper_funcs_glob.src.calc_min_bound_dists.calc_min_bound_dists(trajectory=trajectory,
                                                                                bound1=bound_r_check,
                                                                                bound2=bound_l_check,
                                                                                l_veh_real=veh_dims["l_veh_real"],
                                                                                w_veh_real=veh_dims["w_veh_real"])

    # calculate overall minimum distance
    min_dist = np.amin(min_dists)

    # warn if distance falls below a safety margin of 1.0 m
    if min_dist < 1.0:
        print(
            "WARNING: Minimum distance to boundaries is estimated to %.2fm. Keep in mind that the distance can also"
            " lie on the outside of the track!" % min_dist)
    elif debug:
        print(
            "Minimum distance to boundaries is estimated to %.2fm. Keep in mind that the distance can also lie on the"
            " outside of the track!" % min_dist)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK FINAL TRAJECTORY FOR MAXIMUM CURVATURE AND ACCELERATIONS ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check maximum (absolute) curvature (with a small buffer)
    if np.amax(np.abs(trajectory[:, 4])) > 0.13:
        print("WARNING: Curvature limit is exceeded: %.3frad/m" % np.amax(np.abs(trajectory[:, 4])))

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1.0, trajectory[:, 4],
                             out=np.full(trajectory.shape[0], np.inf),
                             where=trajectory[:, 4] != 0))

    # check max. lateral accelerations (with a small buffer)
    ay_profile = np.divide(np.power(trajectory[:, 5], 2), radii)

    if np.amax(ay_profile) > np.amax(np.abs(ggv[:, 4])) + 1.0:
        print("WARNING: Lateral acceleration limit is exceeded: %.2fm/s2" % np.amax(ay_profile))

    # check max. longitudinal accelerations (with a small buffer)
    if np.amax(trajectory[:, 6]) > np.amax(ggv[:, 2]) + 1.0:
        print("WARNING: Longitudinal acceleration limit (positive) is exceeded: %.2fm/s2"
              % np.amax(trajectory[:, 6]))

    if np.amin(trajectory[:, 6]) < np.amin(ggv[:, 3]) - 1.0:
        print("WARNING: Longitudinal acceleration limit (negative) is exceeded: %.2fm/s2"
              % np.amin(trajectory[:, 6]))

    # check total acceleration (with a small buffer)
    a_tot = np.sqrt(np.power(trajectory[:, 6], 2) + np.power(ay_profile, 2))

    if np.amax(a_tot) > np.amax(np.abs(ggv[:, 2:])) + 1.0:
        print("WARNING: Total acceleration limit is exceeded: %.2fm/s2" % np.amax(a_tot))

    return bound_r, bound_l


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
