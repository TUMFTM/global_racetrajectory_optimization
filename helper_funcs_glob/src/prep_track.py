import numpy as np
import trajectory_planning_helpers as tph
import sys


def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = True,
               min_width: float = None,
               cone_mode: bool = False) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)
    cone_mode:                  DO NOT USE - detect single cone in track widths and inflate region before / after that
                                cone to avoid that the planner is constrainted to the corridor which would lead to
                                curvature oscillations

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # number of consecutive samples required to be larger than min_width in order to finish a cone area
    CONE_DETECTION_AREA_THRESHOLD = 10

    # track_width added to area around cone, that was smaller than min_width [in m]
    CONE_SURROUNDING_EXTRA_INFLATION = 0.5

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # smoothing and interpolating reference track
    reftrack_interp = tph.spline_approximation. \
        spline_approximation(track=reftrack_imp,
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"],
                             debug=debug)

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
        calc_splines(path=refpath_interp_cl,
                     use_dist_scaling=False)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=reftrack_interp,
                                                                         normvec_normalized=normvec_normalized_interp,
                                                                         horizon=10)

    if normals_crossing:
        raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

    # ------------------------------------------------------------------------------------------------------------------
    # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    manipulated_track_width = False

    if min_width is not None:
        # if cone mode (further inflate track right before and after possible cone-location)
        if cone_mode:
            j = -1
            # iterate through every point in track
            for i in range(reftrack_interp.shape[0]):
                if i > j:
                    # get current track-width
                    cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]
                    if cur_width < (min_width + CONE_SURROUNDING_EXTRA_INFLATION * 2):
                        manipulated_track_width = True

                        # cone area detected
                        cone_area = True
                        min_width_detected = np.inf
                        j = i
                        min_j = 0

                        cone_end_cnt = 0

                        # search end of cone-area with threshold and safe index of minimum-track-width
                        while cone_area:
                            cur_width = reftrack_interp[j, 2] + reftrack_interp[j, 3]

                            # if found new minimum width in region (possible cone location)
                            if cur_width < min_width_detected:
                                min_width_detected = cur_width
                                min_j = j

                            # check if still in cone-area
                            if cur_width >= (min_width + CONE_SURROUNDING_EXTRA_INFLATION * 2):
                                cone_end_cnt += 1
                            else:
                                cone_end_cnt = 0

                            if cone_end_cnt > CONE_DETECTION_AREA_THRESHOLD:
                                cone_area = False

                            j += 1

                            # handle start-line overlap
                            if j >= reftrack_interp.shape[0]:
                                j = 0

                        # modify track width in detected area
                        cone_area = True
                        j = i
                        cone_end_cnt = 0
                        while cone_area:
                            cur_width = reftrack_interp[j, 2] + reftrack_interp[j, 3]

                            # inflate to both sides equally (only min.-track-width for cone +/-1, extra infl. else)
                            if min_j - 1 <= j <= min_j + 1:
                                reftrack_interp[j, 2] += (min_width - cur_width) / 2
                                reftrack_interp[j, 3] += (min_width - cur_width) / 2
                            elif cur_width < (min_width + CONE_SURROUNDING_EXTRA_INFLATION * 2):
                                reftrack_interp[j, 2] += (min_width - cur_width) / 2 + CONE_SURROUNDING_EXTRA_INFLATION
                                reftrack_interp[j, 3] += (min_width - cur_width) / 2 + CONE_SURROUNDING_EXTRA_INFLATION

                            # check if still in cone area
                            if cur_width >= (min_width + CONE_SURROUNDING_EXTRA_INFLATION * 2):
                                cone_end_cnt += 1
                            else:
                                cone_end_cnt = 0

                            if cone_end_cnt > CONE_DETECTION_AREA_THRESHOLD:
                                cone_area = False

                            j += 1

                            # handle start-line overlap
                            if j >= reftrack_interp.shape[0]:
                                j = 0

        # just enforce minimum track width
        else:
            for i in range(reftrack_interp.shape[0]):
                cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]
                if cur_width < min_width:
                    manipulated_track_width = True

                    # inflate to both sides equally
                    reftrack_interp[i, 2] += (min_width - cur_width) / 2
                    reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
              " order to match the requirements!", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
