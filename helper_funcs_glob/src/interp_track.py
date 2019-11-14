import numpy as np
import math


def interp_track(reftrack: np.ndarray,
                 stepsize_approx: float = 1.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Use linear interpolation between track points to create new points with equal distances.

    Inputs:
    reftrack:           array containing the track information that shell be interpolated [x, y, w_tr_right, w_tr_left].
    stepsize_approx:    desired stepsize for the interpolation

    Outputs:
    reftrack_interp:    interpolated reference track (unclosed)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTION BODY ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    reftrack_cl = np.vstack((reftrack, reftrack[0]))

    # calculate element lengths (euclidian distance)
    el_lenghts = np.sqrt(np.sum(np.power(np.diff(reftrack_cl[:, :2], axis=0), 2), axis=1))

    # sum up total distance (from start) to every element
    dists_cum = np.cumsum(el_lenghts)
    dists_cum = np.insert(dists_cum, 0, 0.0)

    # calculate desired lenghts depending on specified stepsize (+1 because last element is included)
    no_points_interp = math.ceil(dists_cum[-1] / stepsize_approx) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

    # interpolate closed track points
    reftrack_interp_cl = np.zeros((no_points_interp, 4))
    reftrack_interp_cl[:, 0] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 0])
    reftrack_interp_cl[:, 1] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 1])
    reftrack_interp_cl[:, 2] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 2])
    reftrack_interp_cl[:, 3] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 3])

    # remove closed points
    reftrack_interp = reftrack_interp_cl[:-1]

    return reftrack_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
