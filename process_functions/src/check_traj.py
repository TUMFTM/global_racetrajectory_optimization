import numpy as np
import helper_funcs_glob

"""
Created by:
Alexander Heilmeier

Created on:
22.10.2018

Documentation:
This function checks the generated trajectory and prints several outputs.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def check_traj(reftrack: np.ndarray,
               reftrack_normvec_normalized: np.ndarray,
               trajectory_opt: np.ndarray,
               ggv: np.ndarray,
               veh_dims: dict,
               debug: bool) -> tuple:

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK VEHICLE EDGES FOR MINIMUM DISTANCE TO TRACK BOUNDARIES -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate boundaries and interpolate them to small stepsizes (currently linear interpolation)
    bound1 = reftrack[:, :2] + reftrack_normvec_normalized * np.expand_dims(reftrack[:, 2], 1)
    bound2 = reftrack[:, :2] - reftrack_normvec_normalized * np.expand_dims(reftrack[:, 3], 1)

    # check boundaries for vehicle edges
    bound1_tmp = np.column_stack((bound1, np.zeros((bound1.shape[0], 2))))
    bound2_tmp = np.column_stack((bound2, np.zeros((bound2.shape[0], 2))))

    bound1_check = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound1_tmp,
                                                                   stepsize_approx=1.0)[0]
    bound2_check = helper_funcs_glob.src.interp_track.interp_track(reftrack=bound2_tmp,
                                                                   stepsize_approx=1.0)[0]

    # calculate minimum distances of every trajectory point to the boundaries
    min_dists = helper_funcs_glob.src.edge_check.edge_check(trajectory=trajectory_opt,
                                                            bound1=bound1_check,
                                                            bound2=bound2_check,
                                                            l_veh_real=veh_dims["l_veh_real"],
                                                            w_veh_real=veh_dims["w_veh_real"])

    # calculate overall minimum distance
    min_dist = np.amin(min_dists)

    # warn if distance falls below a safety margin of 1.0 m
    if min_dist < 1.0:
        print(
            "WARNING: Minimum distance to boundaries is estimated to %.2f m. Keep in mind that the distance can also"
            " lie on the outside of the track!" % min_dist)
    elif debug:
        print(
            "Minimum distance to boundaries is estimated to %.2f m. Keep in mind that the distance can also lie on the"
            " outside of the track!" % min_dist)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK FINAL TRAJECTORY -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check maximum (absolute) curvature (with a small buffer)
    if np.amax(np.abs(trajectory_opt[:, 4])) > 0.13:
        print("WARNING: Curvature limit is exceeded: %.3f radpm" % np.amax(np.abs(trajectory_opt[:, 4])))

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1, trajectory_opt[:, 4], out=np.full(trajectory_opt.shape[0], np.inf),
                             where=trajectory_opt[:, 4] != 0))

    # check max lateral accelerations (with a small buffer)
    ay_profile = np.divide(np.power(trajectory_opt[:, 5], 2), radii)

    if np.amax(ay_profile) > np.amax(np.abs(ggv[:, 4])) + 2.0:
        print("WARNING: Lateral acceleration limit is exceeded: %.2f mps2" % np.amax(ay_profile))

    # check max longitudinal accelerations (with a small buffer)
    if np.amax(trajectory_opt[:, 6]) > np.amax(ggv[:, 2]) + 2.0:
        print("WARNING: Longitudinal acceleration limit (positive) is exceeded: %.2f mps2" % np.amax(
            trajectory_opt[:, 6]))

    if np.amin(trajectory_opt[:, 6]) < np.amin(ggv[:, 3]) - 2.0:
        print("WARNING: Longitudinal acceleration limit (negative) is exceeded: %.2f mps2" % np.amin(
            trajectory_opt[:, 6]))

    # check Kamm'scher Kreis (with a small buffer)
    a_tot = np.sqrt(np.power(trajectory_opt[:, 6], 2) + np.power(ay_profile, 2))

    if np.amax(a_tot) > np.amax(np.abs(ggv[:, 2:])) + 2.0:
        print("WARNING: Total acceleration limit is exceeded: %.2f mps2" % np.amax(a_tot))

    return bound1, bound2


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
