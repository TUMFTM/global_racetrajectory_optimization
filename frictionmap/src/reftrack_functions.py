import numpy as np
import math
import matplotlib.pyplot as plt

"""
Created by:
Leonhard Hermansdorfer

Created on:
20.12.2018
"""


def load_reftrack(path2track: str) -> np.array:
    """
    Documentation
    This function loads the track file.

    Input
    :param path2track:              absolute path to reference track file

    Output
    :return reftrack                reference track containing x-,y-coordinates and trackwidths to the right and left of
                                    the reference line [x_m, y_m, trackwidth_right_m, trackwidth_left_m]
    """

    with open(path2track, 'r') as fh:
        reftrack = np.genfromtxt(fh, delimiter=',')

    return reftrack


def check_isclosed_refline(refline: np.ndarray) -> bool:
    """
    Documentation
    This function checks whether the given reference line is a closed or open circuit.

    Input
    :param refline:                 reference line [x_m, y_m]

    Output
    :return bool_isclosed_refline   boolean indicating whether the track is closed / a circuit (=True) or not (=False)
    """

    # user input
    max_dist_isclosed = 10.0  # [m]

    # determine if reference track is expected as closed loop
    dist_last2first = math.sqrt((refline[-1, 0] - refline[0, 0]) ** 2 + (refline[-1, 1] - refline[0, 1]) ** 2)

    # if track is closed, add first row of reference line at the bottom. Otherwise, extrapolate last coordinate and add
    if dist_last2first <= max_dist_isclosed:
        bool_isclosed_refline = True

    else:
        bool_isclosed_refline = False

    return bool_isclosed_refline


def calc_trackboundaries(reftrack: np.ndarray) -> tuple:
    """
    Documentation
    This function calculates the actual coordinates of both track boundaries specified by the reference line and the
    corresponding trackwidths.

    Input
    :param reftrack:                reference track [x_m, y_m, trackwidth_right_m, trackwidth_left_m]

    Output
    :return track_boundary_right    x-,y-coordinates of right trackboundary (from reference line in driving direction)
    :return track_boundary_left     x-,y-coordinates of left trackboundary (from reference line in driving direction)
    """

    refline_normvecs = calc_refline_normvecs(refline=reftrack[:, :2])
    track_boundary_right = reftrack[:, :2] + refline_normvecs[:, :2] * np.expand_dims(reftrack[:, 2], axis=1)
    track_boundary_left = reftrack[:, :2] - refline_normvecs[:, :2] * np.expand_dims(reftrack[:, 3], axis=1)

    return track_boundary_right, track_boundary_left


def calc_refline_normvecs(refline: np.ndarray) -> np.array:
    """
    Documentation
    This function calculates the normal vectors of the reference line at each coordinate (pointing towards the right in
    the direction of driving).

    Input
    :param refline:                 reference line [x_m, y_m]

    Output
    :return refline_normvecs        reference line normal vectors [x_m, y_m]
    """

    bool_isclosed_refline = check_isclosed_refline(refline=refline)

    if bool_isclosed_refline:
        refline = np.vstack((refline[-1], refline, refline[0]))

    refline_grad = np.gradient(refline[:, :3], axis=0)

    # z-vector for calculating cross product to get normal vector
    z = np.array([0.0, 0.0, 1.0])

    refline_crossproduct = np.cross(refline_grad, z)

    norm_factors = np.divide(1.0, np.linalg.norm(refline_crossproduct, axis=1))

    refline_normvecs = refline_crossproduct * norm_factors[:, None]

    if bool_isclosed_refline:
        refline_normvecs = np.delete(refline_normvecs, 0, axis=0)
        refline_normvecs = np.delete(refline_normvecs, -1, axis=0)

    return refline_normvecs


def plot_refline(reftrack: np.ndarray) -> None:
    """
    Documentation
    This function plots the reference line and its normal vectors at each coordinate.

    Input
    :param reftrack:    reference track [x_m, y_m, trackwidth_right_m, trackwidth_left_m]

    Output
    ---
    """

    # get normal vectors
    refline_normvecs = calc_refline_normvecs(refline=reftrack[:, :2])

    # calculate track boundaries
    plt.figure()
    plt.plot(reftrack[:, 0], reftrack[:, 1])

    for row in range(0, refline_normvecs.shape[0]):
        plt.plot([reftrack[row, 0],
                  reftrack[row, 0] + (-refline_normvecs[row, 0] * reftrack[row, 3])],
                 [reftrack[row, 1],
                  reftrack[row, 1] + (-refline_normvecs[row, 1] * reftrack[row, 3])], 'g')

        plt.plot([reftrack[row, 0],
                  reftrack[row, 0] + (refline_normvecs[row, 0] * reftrack[row, 2])],
                 [reftrack[row, 1],
                  reftrack[row, 1] + (refline_normvecs[row, 1] * reftrack[row, 2])], 'r')

    plt.grid()
    plt.title('Reference line and normal vectors')
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')
    plt.axis('equal')

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
