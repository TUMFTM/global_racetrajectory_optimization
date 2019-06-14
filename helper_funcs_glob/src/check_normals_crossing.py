import numpy as np
import trajectory_planning_helpers as tph


def check_normals_crossing(reftrack: np.ndarray,
                           normvec_normalized: np.ndarray) -> bool:
    """
    Created by:
    Alexander Heilmeier

    Created on:
    27.04.2018

    Documentation:
    Check spline normals for crossing points. Returns True if a crossing was found, otherwise False.

    Attention: Can take a while!

    Inputs:
    reftrack:               array containing the reference track [x, y, w_tr_right, w_tr_left]
    normvec_normalized:     array containing normalized normal vectors for every traj. point [x_component, y_component]

    Outputs:
    found_crossing:         bool value showing if a crossing was found or not
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LOOP THROUGH ALL NORMALS AND CHECK CROSSING ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    found_crossing = False
    les_mat = np.zeros((2, 2))  # matrix used for the linear equation system (LES) later
    no_points = reftrack.shape[0]

    for i in range(no_points):

        # check collinearity to any other normal vector -> if collinear then do not use it
        ind_rel_comp = []  # relevant indices of compared normal vectors

        for j, cur_comp_vec in enumerate(normvec_normalized):
            if not np.isclose(np.cross(normvec_normalized[i], cur_comp_vec), 0.0):
                ind_rel_comp.append(j)

        # check crossings
        for ind_comp_cur in ind_rel_comp:

            # LES: x_1 + lambda_1 * nx_1 = x_2 + lambda_2 * nx_2; y_1 + lambda_1 * ny_1 = y_2 + lambda_2 * ny_2;
            const = reftrack[ind_comp_cur, :2] - reftrack[i, :2]
            les_mat[:, 0] = normvec_normalized[i]
            les_mat[:, 1] = -normvec_normalized[ind_comp_cur]

            # solve LES
            lambdas = np.linalg.solve(les_mat, const)

            # we have a crossing within the relevant part if both lambdas lie between -w_tr_left and w_tr_right
            if -reftrack[i, 3] <= lambdas[0] <= reftrack[i, 2] \
                    and -reftrack[ind_comp_cur, 3] <= lambdas[1] <= reftrack[ind_comp_cur, 2]:
                found_crossing = True
                break

        # break if crossing was found
        if found_crossing:
            break

        # print progressbar
        tph.progressbar.progressbar(i=i,
                                    i_total=no_points,
                                    prefix="Normal vector crossing check progress:")

    return found_crossing


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
