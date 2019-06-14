import numpy as np
import trajectory_planning_helpers

"""
Created by:
Alexander Heilmeier

Created on:
22.10.2018

Documentation:
This function includes the algorithm part connected to the interpolation of the raceline after the optimization.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def interp_raceline(stepsize_interp_after_opt: float,
                    refline_interp: np.ndarray,
                    alpha_opt: np.ndarray,
                    normvec_normalized_interp: np.ndarray) -> tuple:

    # calculate minimum curvature line points based on optimized alpha values
    raceline = refline_interp + np.expand_dims(alpha_opt, 1) * normvec_normalized_interp
    raceline_cl = np.vstack((raceline, raceline[0]))

    # calculate new splines with raceline data (required for a better than linear interpolation)
    coeffs_x_opt, coeffs_y_opt, a_opt, normvec_normalized_opt = trajectory_planning_helpers.calc_splines.\
        calc_splines(path=raceline_cl,
                     use_dist_scaling=False)

    # calculate new spline lenghts
    spline_lengths_opt = trajectory_planning_helpers.calc_spline_lengths. \
        calc_spline_lengths(coeffs_x=coeffs_x_opt,
                            coeffs_y=coeffs_y_opt)

    # calculate total track length
    s_total = float(np.sum(spline_lengths_opt))

    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp = trajectory_planning_helpers.\
        interp_splines.interp_splines(spline_lengths=spline_lengths_opt,
                                      coeffs_x=coeffs_x_opt,
                                      coeffs_y=coeffs_y_opt,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp_after_opt)

    # calculate element lenghts
    el_lengths_opt_interp = np.diff(s_points_opt_interp)
    el_lengths_opt_interp_cl = np.append(el_lengths_opt_interp, s_total - s_points_opt_interp[-1])

    return raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, \
           s_points_opt_interp, spline_lengths_opt, el_lengths_opt_interp_cl


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
