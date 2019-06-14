import numpy as np
import quadprog
import time


def opt_shortest_path(reftrack: np.ndarray,
                      normvectors: np.ndarray,
                      w_veh: float,
                      print_debug: bool = False) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Created on:
    15.01.2019

    Documentation:
    This function uses a QP solver to minimize the summed length of a path by moving the path points along their
    normal vectors.

    Please refer to the following paper for further information:
    Braghin, Cheli, Melzi, Sabbioni
    Race Driver Model
    DOI: 10.1016/j.compstruc.2007.04.028

    Inputs:
    reftrack:       array containing the reference track [x, y, w_tr_right, w_tr_left]
    normvectors:    normalized normal vectors for every point of the reference track [x_component, y_component]
    w_veh:          vehicle width in m. It is considered during calculation of the allowed deviations from refer. line.
    print_debug:    bool flag to print debug messages.

    Outputs:
    alpha_shpath:   solution vector of the optimization problem keeping the lateral shift in m for every point.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = reftrack.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    H = np.zeros((no_points, no_points))
    f = np.zeros(no_points)

    for i in range(no_points):
        if i < no_points - 1:
            H[i, i] += 2 * (np.power(normvectors[i, 0], 2) + np.power(normvectors[i, 1], 2))
            H[i, i + 1] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[i + 1, 0]
                                     - 2 * normvectors[i, 1] * normvectors[i + 1, 1])
            H[i + 1, i] = H[i, i + 1]
            H[i + 1, i + 1] = 2 * (np.power(normvectors[i + 1, 0], 2) + np.power(normvectors[i + 1, 1], 2))

            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[i + 1, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[i + 1, 1]
            f[i + 1] = -2 * normvectors[i + 1, 0] * reftrack[i, 0] \
                       - 2 * normvectors[i + 1, 1] * reftrack[i, 1] \
                       + 2 * normvectors[i + 1, 0] * reftrack[i + 1, 0] \
                       + 2 * normvectors[i + 1, 1] * reftrack[i + 1, 1]

        else:
            H[i, i] += 2 * (np.power(normvectors[i, 0], 2) + np.power(normvectors[i, 1], 2))
            H[i, 0] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[0, 0] - 2 * normvectors[i, 1] * normvectors[0, 1])
            H[0, i] = H[i, 0]
            H[0, 0] += 2 * (np.power(normvectors[0, 0], 2) + np.power(normvectors[0, 1], 2))

            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[0, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[0, 1]
            f[0] += -2 * normvectors[0, 0] * reftrack[i, 0] - 2 * normvectors[0, 1] * reftrack[i, 1] \
                    + 2 * normvectors[0, 0] * reftrack[0, 0] + 2 * normvectors[0, 1] * reftrack[0, 1]

    # ------------------------------------------------------------------------------------------------------------------
    # CALL QUADRATIC PROGRAMMING ALGORITHM -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    quadprog interface description taken from 
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

    Solve a Quadratic Program defined as:

        minimize
            (1/2) * alpha.T * H * alpha + f.T * alpha

        subject to
            G * alpha <= h
            A * alpha == b

    using quadprog <https://pypi.python.org/pypi/quadprog/>.

    Parameters
    ----------
    H : numpy.array
        Symmetric quadratic-cost matrix.
    f : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).

    Returns
    -------
    alpha : numpy.array
            Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    The quadprog solver only considers the lower entries of `H`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # set minimum deviation to zero
    dev_max_right[dev_max_right < 0.001] = 0.001
    dev_max_left[dev_max_left < 0.001] = 0.001

    # consider value boundaries (-dev_max <= alpha <= dev_max)
    G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    h = np.ones(2 * no_points) * np.append(dev_max_right, dev_max_left)

    # save start time
    t_start = time.perf_counter()

    # solve problem
    alpha_shpath = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # print runtime into console window
    if print_debug:
        print("Solver runtime opt_shortest_path: " + "{:.3f}".format(time.perf_counter() - t_start) + " seconds")

    return alpha_shpath


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
