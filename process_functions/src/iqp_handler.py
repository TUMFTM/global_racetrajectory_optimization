import numpy as np
import trajectory_planning_helpers
import opt_geometrical
import helper_funcs_glob

"""
Created by:
Alexander Heilmeier

Created on:
15.01.2019

Documentation:
This function handles the iterative call of the quadratic optimization problem during trajectory optimization.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def iqp_handler(reftrack: np.ndarray,
                normvectors: np.ndarray,
                A: np.ndarray,
                stepsize_reg: float,
                kappa_bound: float,
                w_veh: float,
                print_debug: bool,
                plot_debug: bool,
                iters_min: int,
                curv_error_allowed: float) -> tuple:

    # ------------------------------------------------------------------------------------------------------------------
    # IQP --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set initial data
    reftrack_tmp = reftrack
    normvec_normalized_tmp = normvectors
    A_tmp = A

    # loop (sequential quadratic programming)
    iter_cur = 0

    while True:
        iter_cur += 1

        # calculate intermediate solution and catch sum of squared curvature errors
        alpha_tmp, curv_error_max = opt_geometrical.src.opt_min_curv.\
            opt_min_curv(reftrack=reftrack_tmp,
                         normvectors=normvec_normalized_tmp,
                         A=A_tmp,
                         kappa_bound=kappa_bound,
                         w_veh=w_veh,
                         print_debug=print_debug,
                         plot_debug=plot_debug)

        # print some progress information
        if print_debug:
            print("Minimum curvature IQP: iteration %i, curv_error_max: %.4f radpm" % (iter_cur, curv_error_max))

        # restrict solution space to improve validity of the linearization during the first steps
        if iter_cur < iters_min:
            alpha_tmp *= iter_cur * 1.0 / iters_min

        # check stop conditions: minimum number of iterations and curvature error
        if iter_cur >= iters_min and curv_error_max <= curv_error_allowed:
            if print_debug:
                print("Finished IQP!")
            break

        # calculate new boundaries depending on alpha values
        reftrack_tmp[:, 2] -= alpha_tmp
        reftrack_tmp[:, 3] += alpha_tmp

        # calculate new line points
        reftrack_tmp[:, :2] = reftrack_tmp[:, :2] + np.expand_dims(alpha_tmp, 1) * normvec_normalized_tmp

        # --------------------------------------------------------------------------------------------------------------
        # INTERPOLATION TO EQUAL STEPSIZES -----------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate new splines with raceline data (required for a better than linear interpolation)
        refpath_tmp_cl = np.vstack((reftrack_tmp[:, :2], reftrack_tmp[0, :2]))

        coeffs_x_tmp, coeffs_y_tmp, A_tmp, normvec_normalized_tmp = trajectory_planning_helpers.calc_splines.\
            calc_splines(path=refpath_tmp_cl,
                         use_dist_scaling=False)

        # calculate new spline lenghts
        spline_lengths_tmp = trajectory_planning_helpers.calc_spline_lengths. \
            calc_spline_lengths(coeffs_x=coeffs_x_tmp,
                                coeffs_y=coeffs_y_tmp)

        # interpolate splines for evenly spaced raceline points
        refline_tmp, spline_inds_tmp, t_values_tmp = trajectory_planning_helpers.interp_splines. \
            interp_splines(spline_lengths=spline_lengths_tmp,
                           coeffs_x=coeffs_x_tmp,
                           coeffs_y=coeffs_y_tmp,
                           incl_last_point=False,
                           stepsize_approx=stepsize_reg)[:3]

        # interpolate track widths accordingly
        ws_track_tmp = helper_funcs_glob.src.interp_w_track.interp_w_track(w_track=reftrack_tmp[:, 2:4],
                                                                           spline_inds=spline_inds_tmp,
                                                                           t_vals=t_values_tmp,
                                                                           incl_last_point=False)

        # create new reftrack
        reftrack_tmp = np.column_stack((refline_tmp, ws_track_tmp))

        # --------------------------------------------------------------------------------------------------------------
        # GET ACCORDING SPLINES FOR OPTIMIZATION PROBLEM ---------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate new splines with raceline data (required for a better than linear interpolation)
        refpath_tmp_cl = np.vstack((reftrack_tmp[:, :2], reftrack_tmp[0, :2]))

        coeffs_x_tmp, coeffs_y_tmp, A_tmp, normvec_normalized_tmp = trajectory_planning_helpers.calc_splines.\
            calc_splines(path=refpath_tmp_cl,
                         use_dist_scaling=False)

    return alpha_tmp, reftrack_tmp, normvec_normalized_tmp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
