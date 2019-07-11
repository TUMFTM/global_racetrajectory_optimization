import numpy as np
import math
import quadprog
# import cvxopt
import time
import matplotlib.pyplot as plt


def opt_min_curv(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float,
                 print_debug: bool = False,
                 plot_debug: bool = False) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Created on:
    23.01.2018

    Documentation:
    This function uses a QP solver to minimize the summed curvature of a path by moving the path points along their
    normal vectors.

    Please refer to our paper for further information:
    Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann
    Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar
    DOI: 10.1080/00423114.2019.1631455

    Hint: CVXOPT can be used instead of quadprog by uncommenting the import and corresponding code section.

    Inputs:
    reftrack:       array containing the reference track [x, y, w_tr_right, w_tr_left]
    normvectors:    normalized normal vectors for every point of the reference track [x_component, y_component]
    A:              linear equation system matrix for splines in x and y direction
                        -> System matrices have the form a_i, b_i * t, c_i * t^2, d_i * t^3
    kappa_bound:    curvature boundary to keep during optimization.
    w_veh:          vehicle width in m. It is considered during calculation of the allowed deviations from refer. line.
    print_debug:    bool flag to print debug messages.
    plot_debug:     bool flag to plot the curvatures that are calculated based on the original linearization and on a
                        linearization around the solution.

    Outputs:
    alpha_mincurv:  solution vector of the optimization problem keeping the lateral shift in m for every point.
    curv_error_max: maximum curvature error when comparing the curvature calculated on the basis of the linearization
                    around the original refererence track and around the solution.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = reftrack.shape[0]

    # create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
    # information
    A_ex_b = np.zeros((no_points, no_points * 4), dtype=np.int)

    for i in range(no_points):
        A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
    # information
    A_ex_c = np.zeros((no_points, no_points * 4), dtype=np.int)

    for i in range(no_points):
        A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_points * 4, no_points))
    M_y = np.zeros((no_points * 4, no_points))

    for i in range(no_points):
        j = i * 4

        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]  # close spline

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    # set up q_x and q_y matrices including the point coordinate information
    q_x = np.zeros((no_points * 4, 1))
    q_y = np.zeros((no_points * 4, 1))

    for i in range(no_points):
        j = i * 4

        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    # set up P_xx, P_xy, P_yy matrices
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # divide where not zero (diag elements)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # make H symmetric

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)   # remove non-singleton dimensions

    # ------------------------------------------------------------------------------------------------------------------
    # KAPPA CONSTRAINTS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # original curvature part (static part)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

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

    # check that there is space remaining between left and right maximum deviation (both can be negative as well!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise ValueError("Problem not solvable, track might be too small to run with current safety distance!")

    # consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # save start time
    t_start = time.perf_counter()

    # solve problem (CVXOPT) -------------------------------------------------------------------------------------------
    # args = [cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(G), cvxopt.matrix(h)]
    # sol = cvxopt.solvers.qp(*args)
    #
    # if 'optimal' not in sol['status']:
    #     print("WARNING: Optimal solution not found!")
    #
    # alpha_mincurv = np.array(sol['x']).reshape((H.shape[1],))

    # solve problem (quadprog) -----------------------------------------------------------------------------------------
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # print runtime into console window
    if print_debug:
        print("Solver runtime opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + " seconds")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CURVATURE ERROR ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate curvature once based on original linearization and once based on a new linearization around the solution
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

    curv_orig_lin = np.zeros(no_points)
    curv_sol_lin = np.zeros(no_points)

    for i in range(no_points):
        curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                          / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
        curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                           / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)

    if plot_debug:
        plt.plot(curv_orig_lin)
        plt.plot(curv_sol_lin)
        plt.legend(("original linearization", "solution based linearization"))
        plt.show()

    # calculate maximum curvature error
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

    return alpha_mincurv, curv_error_max


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
