import numpy as np
import helper_funcs_glob
import trajectory_planning_helpers

"""
Created by:
Alexander Heilmeier

Created on:
22.10.2018

Documentation:
This function prepares the inserted reference track for optimization.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def prep_track(reftrack_imp: np.ndarray,
               pars: dict,
               debug: bool,
               check_normal_crossings: bool) -> tuple:

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    reftrack_interp = trajectory_planning_helpers.spline_approximation. \
        spline_approximation(track=reftrack_imp,
                             k_reg=pars["reg_smooth_opts"]["k_reg"],
                             s_reg=pars["reg_smooth_opts"]["s_reg"],
                             stepsize_prep=pars["stepsizes"]["stepsize_prep"],
                             stepsize_reg=pars["stepsizes"]["stepsize_reg"],
                             debug=debug)

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = trajectory_planning_helpers.calc_splines.\
        calc_splines(path=refpath_interp_cl,
                     use_dist_scaling=False)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if check_normal_crossings:
        normals_crossing = helper_funcs_glob.src.check_normals_crossing.\
            check_normals_crossing(reftrack=reftrack_interp, normvec_normalized=normvec_normalized_interp)

        if normals_crossing:
            raise IOError("At least two spline normals have a crossing point!")

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
