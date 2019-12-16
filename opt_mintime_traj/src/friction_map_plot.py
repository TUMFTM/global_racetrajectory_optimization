import numpy as np
import math
import json
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def friction_map_plot(filepath_tpamap: str,
                      filepath_tpadata: str,
                      filepath_referenceline: str):
    """
    Created by:
    Leonhard Hermansdorfer

    Documentation:
    Function to visualize the friction map data and the reference line for a given race track.

    The friction map is located in "/inputs/frictionmaps/TRACKNAME_tpamap.csv"
    The fricton map data is located in "/inputs/frictionmaps/TRACKNAME_tpadata.json"
    The reference line is located in "/inputs/tracks/TRACKNAME.csv"

    Inputs:
    filepath_tpamap:            path to friction map representing the race track of interest (*_tpamap.csv)
    filepath_tpadata:           path to corresponding friction data of the above specified map (*_tpadata.json)
    filepath_referenceline:     path to corresponding reference line of the above specified friction map
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD DATA FROM FILES ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    print('INFO: Loading friction map data...')

    # load reference line from csv-file
    referenceline = np.loadtxt(filepath_referenceline, comments='#', delimiter=';')

    # load friction map (tpamap) from csv-file
    map_coordinates = np.loadtxt(filepath_tpamap, comments='#', delimiter=';')
    tpamap_loaded = cKDTree(map_coordinates)

    # load friction data corresponding to the chosen friction map
    with open(filepath_tpadata, 'r') as fh:
        tpadata_dict_string = json.load(fh)
    tpadata_loaded = {int(k): np.asarray(v) for k, v in tpadata_dict_string.items()}

    print('INFO: Friction map data loaded successfully!')

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE DATA FOR PLOTTING ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    print('INFO: Preprocessing friction map data for visualization... (takes some time)')

    # read values from dict
    list_coord = tpamap_loaded.data[tpamap_loaded.indices]

    list_mue = []

    for idx in tpamap_loaded.indices:
        list_mue.append(tpadata_loaded[idx])

    # recalculate stepsize of friction map
    x_stepsize = abs(list_coord[1, 0] - list_coord[0, 0])
    y_stepsize = abs(list_coord[1, 1] - list_coord[0, 1])

    # load coordinate values from friction map
    tree_points = tpamap_loaded.data

    # determine min/max of coordinate values in both directions to set up 2d array for countourf plotting
    x_min = math.floor(np.amin(tree_points[:, 0]))
    x_max = math.ceil(np.amax(tree_points[:, 0]))

    y_min = math.floor(np.amin(tree_points[:, 1]))
    y_max = math.ceil(np.amax(tree_points[:, 1]))

    x_vals = np.arange(x_min - (20.0 * x_stepsize), x_max + (19.0 * x_stepsize), x_stepsize)
    y_vals = np.arange(y_min - (20.0 * y_stepsize), y_max + (19.0 * y_stepsize), y_stepsize)

    # set up an empty 2d array which is then filled wiich corresponding mue values
    z = np.full((y_vals.shape[0], x_vals.shape[0]), fill_value=np.nan)

    # plot 2D array
    for row, mue in zip(list_coord, list_mue):
        index_column = int((row[0] - min(x_vals)) / x_stepsize)
        index_row = int((-row[1] + max(y_vals)) / y_stepsize)

        z[index_row, index_column] = mue

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE PLOT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    print('INFO: Plotting friction map data...')

    plt.figure()

    # plot reference line
    plt.plot(referenceline[:, 0], referenceline[:, 1], 'r')

    # set axis limits
    plt.xlim(min(referenceline[:, 0]) - 100.0, max(referenceline[:, 0]) + 100.0)
    plt.ylim(min(referenceline[:, 1]) - 100.0, max(referenceline[:, 1]) + 100.0)

    # plot 2D contour representing the friction data (mue-values)
    plt.contourf(x_vals, np.flipud(y_vals), z)

    # set up plot
    plt.colorbar(label='mue-values')
    plt.title('mue-values of the racetrack')
    plt.xlabel("east in m")
    plt.ylabel("north in m")
    plt.axis('equal')

    plt.show()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import os.path

    module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tpamap_path = os.path.join(module_path, 'inputs', 'frictionmaps', 'berlin_2018_tpamap.csv')
    tpadata_path = os.path.join(module_path, 'inputs', 'frictionmaps', 'berlin_2018_varmue08-12_tpadata.json')
    referenceline_path = os.path.join(module_path, 'inputs', 'tracks', 'berlin_2018.csv')

    friction_map_plot(filepath_tpamap=tpamap_path,
                      filepath_tpadata=tpadata_path,
                      filepath_referenceline=referenceline_path)
