import numpy as np


def interp_w_track(w_track: np.ndarray,
                   spline_inds: np.ndarray,
                   t_vals: np.ndarray,
                   incl_last_point: bool = False) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Created on:
    06.06.2019

    Documentation:
    The function (linearly) interpolates the track widths in the same steps as the splines were interpolated before.

    Inputs:
    w_track:            array containing the track widths [w_tr_right, w_tr_left]
    spline_inds:        indices that show which spline (and here w_track element) shell be interpolated.
    t_vals:             t values on the spline specified by spline_inds
    incl_last_point:    bool flag to show if last point should be included or not.

    All inputs are unclosed.

    Outputs:
    w_track_interp:     array with interpolated track widths.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    w_track_cl = np.vstack((w_track, w_track[0]))
    no_interp_points = t_vals.size  # unclosed

    if incl_last_point:
        w_track_interp = np.zeros((no_interp_points + 1, 2))
        w_track_interp[-1] = w_track_cl[-1]
    else:
        w_track_interp = np.zeros((no_interp_points, 2))

    # loop through every interpolation point
    for i in range(no_interp_points):
        # find the spline that hosts the current interpolation point
        ind_spl = spline_inds[i]

        # calculate track widths (linear approximation assumed along one spline)
        w_track_interp[i, 0] = np.interp(t_vals[i], (0.0, 1.0), w_track_cl[ind_spl:ind_spl+2, 0])
        w_track_interp[i, 1] = np.interp(t_vals[i], (0.0, 1.0), w_track_cl[ind_spl:ind_spl+2, 1])

    return w_track_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
