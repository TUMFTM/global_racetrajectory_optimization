import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os.path
import json
import frictionmap

"""
Created by:
Leonhard Hermansdorfer

Created on:
01.02.2019

Documentation:
To plot friction map data from an already existing file, adjust trackname and filenames at the bottom of this file and 
run this file directly.
"""


def plot_tpamap_fromFile(track_name: str,
                         filename_tpamap: str,
                         filename_frictiondata: str) -> None:
    """
    Documentation
    This function loads the friction map file (*_tpamap.csv') and the friction data file ('*_tpadata.json') and creates
    all variables necessary for plotting the friction map as a grid containing the corresponding friction data.

    Input
    :param track_name:              name of the race track
    :param filename_tpamap:         filename of the file containing the friction map ('*_tpamap.csv'')
    :param filename_frictiondata:   filename of the file containing the friction data ('*_tpadata.json')

    Output
    ---
    """

    # Path Management --------------------------------------------------------------------------------------------------

    path2module = os.path.dirname(os.path.abspath(__file__)).split('frictionmap')[0]
    path2reftrack_file = os.path.join(path2module, 'inputs', 'tracks', track_name + '.csv')
    filepath_frictionmap = os.path.join(path2module, 'inputs', 'frictionmaps', filename_tpamap)
    filepath_frictiondata = os.path.join(path2module, 'inputs', 'frictionmaps', filename_frictiondata)

    # Read Files -------------------------------------------------------------------------------------------------------

    # load reference line and calculate track boundaries
    reftrack = frictionmap.src.reftrack_functions.load_reftrack(path2track=path2reftrack_file)
    trackbound_right, trackbound_left = frictionmap.src.reftrack_functions.calc_trackboundaries(reftrack=reftrack)

    # load friction map
    with open(filepath_frictionmap, 'rb') as fh:
        map_coordinates = np.loadtxt(fh, comments='#', delimiter=';')
        tpamap_loaded = cKDTree(map_coordinates)

    # load friction data
    with open(filepath_frictiondata, 'r') as fh:
        tpadata_dict_string = json.load(fh)
        tpadata_loaded = {int(k): np.asarray(v) for k, v in tpadata_dict_string.items()}

    # call function to plot friction map and corresponding friction data
    plot_tpamap_fromVariable(tpa_map=tpamap_loaded,
                             tpa_data=tpadata_loaded,
                             refline=reftrack[:, :2],
                             trackbound_right=trackbound_right,
                             trackbound_left=trackbound_left)


def plot_tpamap_fromVariable(tpa_map: cKDTree,
                             tpa_data: dict,
                             refline: np.array,
                             trackbound_right: np.array,
                             trackbound_left: np.array) -> None:
    """
    Documentation
    This function plots the friction map as a grid without the corresponding friction data.

    Input
    :param tpa_map:             cKDTree object containing the coordinates of the friction map
    :param tpa_data:            dictionary containing the friction data for each grid cell of the friction map
    :param refline:             array consisting of the x-,y-coordinates of the reference line
    :param trackbound_right:    array consisting of the x-,y-coordinates of the right track boundary
    :param trackbound_left:     array consisting of the x-,y-coordinates of the left track boundary

    Output
    ---
    """

    print("INFO: Plotting friction map with data...")

    list_mue = []
    list_coord = []

    # read values from dict
    for index in tpa_map.indices:
        list_coord.append(tpa_map.data[index])
        list_mue.append(tpa_data[index])

    list_coord = np.array(list_coord)

    # recalculate width of grid cells of friction map (width is set by the user during map generation)
    cellwidth_m = max(abs(tpa_map.data[0] - tpa_map.data[1]))

    plt.figure()
    plt.plot(refline[:, 0], refline[:, 1], 'r')
    plt.plot(trackbound_left[:, 0], trackbound_left[:, 1], 'b')
    plt.plot(trackbound_right[:, 0], trackbound_right[:, 1], 'b')

    plt.axis('equal')
    plt.xlim(np.amin(refline[:, 0]) - 100.0, np.amax(refline[:, 0]) + 100.0)
    plt.ylim(np.amin(refline[:, 1]) - 100.0, np.amax(refline[:, 1]) + 100.0)

    # create contourf plot ---------------------------------------------------------------------------------------------

    x_min = math.floor(min(tpa_map.data[:, 0]))
    x_max = math.ceil(max(tpa_map.data[:, 0]))

    y_min = math.floor(min(tpa_map.data[:, 1]))
    y_max = math.ceil(max(tpa_map.data[:, 1]))

    x_vals = np.arange(x_min - 10.0, x_max + 9.5, cellwidth_m)
    y_vals = np.arange(y_min - 10.0, y_max + 9.5, cellwidth_m)

    z = np.full((y_vals.shape[0], x_vals.shape[0]), np.nan)

    for row, mue in zip(list_coord, list_mue):
        index_column = int((row[0] - min(x_vals)) / cellwidth_m)
        index_row = int((-1 * row[1] + max(y_vals)) / cellwidth_m)

        z[index_row, index_column] = mue

    # change colorbar settings when only a single mue value is set globally
    if min(list_mue) == max(list_mue):
        con = plt.contourf(x_vals, np.flipud(y_vals), z, 1)
        cbar_tickrange = np.asarray(min(list_mue))
        cbar_label = 'global mue value = ' + str(min(list_mue)[0])

    else:
        con = plt.contourf(x_vals, np.flipud(y_vals), z,
                           np.arange(np.round(min(list_mue) - 0.05, 1), np.round(max(list_mue) + 0.06, 1) + 0.01, 0.02))
        cbar_tickrange = np.arange(np.round(min(list_mue) - 0.05, 1), np.round(max(list_mue) + 0.06, 1) + 0.01, 0.05)
        cbar_label = 'local mue values'

    # create a colorbar for the ContourSet returned by the contourf call
    cbar = plt.colorbar(con, cmap='viridis')
    cbar.set_ticks(cbar_tickrange.round(2).tolist())
    cbar.set_label(cbar_label)

    plt.title('friction map and data')
    plt.xlabel('x in meters')
    plt.ylabel('y in meters')

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    track_name = 'berlin_2018'
    filename_tpamap = 'berlin_2018_tpamap.csv'
    filename_tpadata = 'berlin_2018_varmue08-12_tpadata.json'

    plot_tpamap_fromFile(track_name, filename_tpamap, filename_tpadata)
