import numpy as np
import math
import json
import time
from datetime import datetime
import os.path
import matplotlib.path as mplPath
from scipy.spatial import cKDTree
import frictionmap

"""
Created by:
Leonhard Hermansdorfer

Created on:
01.12.2018

Documentation:
This script generates a grid respresenting the friction map with specified cellwidth. Additionally, it fills the
corresponding cells of the friction map with a default mue value.
"""

# ----------------------------------------------------------------------------------------------------------------------
# USER INPUT -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# track_name:             track name for which a friction map should be generated (csv must be in the inputs folder).
# initial_mue:            mue value which should be used to initialize the friction map (all cells contain this value).
# cellwidth_m:            width of the grid cells of the friction map (cells are quadratic).
# inside_trackbound:      specifies which trackbound is on the inside of the racetrack ('right' or 'left'). This is
#                         only necessary for circuits (closed racetracks).
# bool_show_plots:        boolean which enables plotting of the reference line, the friction map and the corresponding
#                         mue values

track_name = "modena_2019"
initial_mue = 0.8
cellwidth_m = 2.0
inside_trackbound = 'right'
bool_show_plots = True

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION -------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# determine names of output files
datetime_save = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_tpamap = datetime_save + '_' + track_name + '_tpamap.csv'
filename_tpadata = datetime_save + '_' + track_name + '_tpadata.json'

# set paths
path2module = os.path.dirname(os.path.abspath(__file__))
path2reftrack_file = os.path.join(path2module, 'inputs', 'tracks', track_name + '.csv')
path2tpamap_file = os.path.join(path2module, 'inputs', 'frictionmaps', filename_tpamap)
path2tpadata_file = os.path.join(path2module, 'inputs', 'frictionmaps', filename_tpadata)

# ----------------------------------------------------------------------------------------------------------------------
# CALCULATE REFERENCE LINE ---------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# read reference track file
reftrack = frictionmap.src.reftrack_functions.load_reftrack(path2track=path2reftrack_file)

# check whether reference line is closed (is the race track a circuit or not)
bool_isclosed_refline = frictionmap.src.reftrack_functions.check_isclosed_refline(refline=reftrack[:, :2])

# calculate coordinates of the track boundaries
reftrackbound_right, reftrackbound_left = frictionmap.src.reftrack_functions.calc_trackboundaries(reftrack=reftrack)

# ----------------------------------------------------------------------------------------------------------------------
# SAMPLE COORDINATES FOR FRICTION MAP ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

timer_start = time.perf_counter()

# match left/right track boundary to "inside" and "outside" (only necessary for circuits)
if bool_isclosed_refline and inside_trackbound == 'right':
    trackbound_inside = reftrackbound_right
    trackbound_outside = reftrackbound_left

    sign_trackbound = -1

else:
    trackbound_inside = reftrackbound_left
    trackbound_outside = reftrackbound_right

    sign_trackbound = 1

# set a default distance which is added / subtracted from max / min reference line coordinate to ensure that whole
# racetrack is covered during coordinate point sampling
default_delta = int(math.ceil(np.amax(reftrack[:, 2]) + np.amax(reftrack[:, 3]) + 5.0))

# calculate necessary range to cover the whole racetrack with grid xyRange = [x_min, x_max, y_min, y_max]
xyRange = [int(math.floor(np.amin(reftrack[:, 0]) - default_delta)),
           int(math.ceil(np.amax(reftrack[:, 0]) + default_delta)),
           int(math.floor(np.amin(reftrack[:, 1]) - default_delta)),
           int(math.ceil(np.amax(reftrack[:, 1]) + default_delta))]

# set-up 2D-grid
x_grid = np.arange(xyRange[0], xyRange[1] + 0.1, cellwidth_m)
y_grid = np.arange(xyRange[2], xyRange[3] + 0.1, cellwidth_m)

# get number of coordinates for array initialization
size_array = x_grid.shape[0] * y_grid.shape[0]
coordinates = np.empty((size_array, 2))

# create coordinate array which contains all coordinate of the defined grid
i_row = 0

for x_coord in x_grid:
    coordinates[i_row:i_row + y_grid.shape[0], 0] = np.full((y_grid.shape[0]), x_coord)
    coordinates[i_row:i_row + y_grid.shape[0], 1] = y_grid
    i_row += y_grid.shape[0]

# set maximum distance between grid cells outside the track and trackboundaries to determine all relevant grid cells
dist_to_trackbound = cellwidth_m * 1.1

# distinguish between a closed racetrack (circuit) and an "open" racetrack
if bool_isclosed_refline:
    bool_isIn_rightBound = mplPath.Path(trackbound_outside).\
        contains_points(coordinates, radius=(dist_to_trackbound * sign_trackbound))
    bool_isIn_leftBound = mplPath.Path(trackbound_inside).\
        contains_points(coordinates, radius=-(dist_to_trackbound * sign_trackbound))
    bool_OnTrack = (bool_isIn_rightBound & ~bool_isIn_leftBound)

else:
    trackbound = np.vstack((trackbound_inside, np.flipud(trackbound_outside)))
    bool_OnTrack = mplPath.Path(trackbound).contains_points(coordinates, radius=-dist_to_trackbound)

# generate the friction map with coordinates which are within the trackboundaries or within the defined range outside
tpa_map = cKDTree(coordinates[bool_OnTrack])

print('INFO: Time elapsed for tpa_map building: {:.3f}s\nINFO: tpa_map contains {} coordinate points'.format(
    (time.perf_counter() - timer_start), tpa_map.n))

# ----------------------------------------------------------------------------------------------------------------------
# SAVE FRICTION MAP ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

timer_start = time.perf_counter()

# create dictionary filled with default mue value (value as numpy array)
tpamap_indices = tpa_map.indices
tpa_data = dict(zip(tpamap_indices, np.full((tpamap_indices.shape[0], 1), initial_mue)))

print('INFO: Time elapsed for tpa_data dictionary building: {:.3f}s'.format(time.perf_counter() - timer_start))

# save friction map (only grid) ('*_tpamap.csv')
with open(path2tpamap_file, 'wb') as fh:
    np.savetxt(fh, tpa_map.data, fmt='%0.4f', delimiter=';', header='x_m;y_m')

print('INFO: tpa_map saved successfully!')

# get tpadata as string to save as a dictionary (as .json file)
tpa_data_string = {str(k): list(v) for k, v in tpa_data.items()}

# save friction data ('*_tpadata.json')
with open(path2tpadata_file, 'w') as fh:
    json.dump(tpa_data_string, fh, separators=(',', ': '))

print('INFO: tpa_data saved successfully!')

# ----------------------------------------------------------------------------------------------------------------------
# CREATE PLOTS ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if bool_show_plots:
    # plot reference line and normal vectors
    frictionmap.src.reftrack_functions.plot_refline(reftrack=reftrack)

    # plot spatial grid of friction map
    frictionmap.src.plot_frictionmap_grid.\
        plot_voronoi_fromVariable(tree=tpa_map,
                                  refline=reftrack[:, :2],
                                  trackbound_left=reftrackbound_left,
                                  trackbound_right=reftrackbound_right)

    # plot friction data of friction map
    frictionmap.src.plot_frictionmap_data.\
        plot_tpamap_fromVariable(tpa_map=tpa_map,
                                 tpa_data=tpa_data,
                                 refline=reftrack[:, :2],
                                 trackbound_left=reftrackbound_left,
                                 trackbound_right=reftrackbound_right)
