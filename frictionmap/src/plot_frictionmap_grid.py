import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree
import os.path
import frictionmap

"""
Created by:
Leonhard Hermansdorfer

Created on:
01.12.2018

Documentation:
To plot a friction map from an already existing file, adjust trackname and filenames at the bottom of this file and 
run this file directly.
"""


def plot_voronoi_fromFile(track_name: str,
                          filename_frictionmap: str) -> None:
    """
    Documentation
    This function loads the friction map file (*_tpamap.csv') and creates all variables necessary for plotting the
    friction map as a grid without the corresponding friction data.

    Input
    :param track_name:              name of the race track
    :param filename_frictionmap:    filename of the file containing the friction map ('*_tpamap.csv'')

    Output
    ---
    """

    # Path Management --------------------------------------------------------------------------------------------------

    path2module = os.path.dirname(os.path.abspath(__file__)).split('frictionmap')[0]
    path2reftrack_file = os.path.join(path2module, 'inputs', 'tracks', track_name + '.csv')
    filepath_frictionmap = os.path.join(path2module, 'inputs', 'frictionmaps', filename_frictionmap)

    # Read Files -------------------------------------------------------------------------------------------------------

    # load reference line and calculate track boundaries
    reftrack = frictionmap.src.reftrack_functions.load_reftrack(path2track=path2reftrack_file)
    trackbound_right, trackbound_left = frictionmap.src.reftrack_functions.calc_trackboundaries(reftrack=reftrack)

    # load friction map
    with open(filepath_frictionmap, 'rb') as fh:
        map_coordinates = np.loadtxt(fh, comments='#', delimiter=';')
        tpamap_loaded = cKDTree(map_coordinates)

    # call function to plot friction map
    plot_voronoi_fromVariable(tree=tpamap_loaded,
                              refline=reftrack[:, :2],
                              trackbound_right=trackbound_right,
                              trackbound_left=trackbound_left)


def plot_voronoi_fromVariable(tree: cKDTree,
                              refline: np.array,
                              trackbound_right: np.array,
                              trackbound_left: np.array,) -> None:
    """
    Documentation
    This function plots the friction map as a grid without the corresponding friction data.

    Input
    :param tree:                cKDTree object containing the coordinates of the friction map
    :param refline:             array consisting of the x-,y-coordinates of the reference line
    :param trackbound_right:    array consisting of the x-,y-coordinates of the right track boundary
    :param trackbound_left:     array consisting of the x-,y-coordinates of the left track boundary

    Output
    ---
    """

    print("INFO: Plotting friction map grid - 2 plots...")

    # plot 1
    tree_points = tree.data

    plt.figure()
    plt.scatter(tree_points[:, 0], tree_points[:, 1])
    plt.plot(refline[:, 0], refline[:, 1], 'r')

    plt.axis('equal')
    plt.title('grid coordinates and reference line')
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')

    plt.show()

    # plot 2
    vor = Voronoi(tree_points[:, 0:2])

    voronoi_plot_2d(vor, show_vertices=False)
    plt.plot(refline[:, 0], refline[:, 1], 'r')
    plt.plot(trackbound_left[:, 0], trackbound_left[:, 1], 'b')
    plt.plot(trackbound_right[:, 0], trackbound_right[:, 1], 'b')

    plt.axis('equal')
    plt.xlim(np.amin(refline[:, 0]) - 100.0, np.amax(refline[:, 0]) + 100.0)
    plt.ylim(np.amin(refline[:, 1]) - 100.0, np.amax(refline[:, 1]) + 100.0)
    plt.title('grid cells, reference line and track boundaries')
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    track_name = 'modena_2019'
    filename_tpamap = 'modena2019_tpamap.csv'

    plot_voronoi_fromFile(track_name, filename_tpamap)
