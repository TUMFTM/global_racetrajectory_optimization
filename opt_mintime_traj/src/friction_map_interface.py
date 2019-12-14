import numpy as np
from scipy.spatial import cKDTree
import json


class FrictionMapInterface:
    """
    Created by:
    Leonhard Hermansdorfer

    Documentation:
    This class loads the friction map (*_tpamap.csv) and the corresponding data (*_tpadata.json) and provides an
    interface to fetch friction data for a requested position on the race track.

    NOTE:           Naming of map and data file has to be consistent! Everything replaced by '*' has to be identical in
                    order to load correct data to a given map.

    The following data must be available for the friction map:
    tpa_map:        csv-file containing the map information (x,y-coordinates of each grid cell;
                    '*_tpamap.csv' located in inputs folder)
    tpa_data:       json-file containing specific data for each grid cell (e.g. coefficient of friction);
                    '*_tpadata.json' located in inputs folder)
    """

    def __init__(self,
                 tpamap_path: str,
                 tpadata_path: str) -> None:

        # load friction map (only contains x,y coordinates and the corresponding grid cell indices) and
        # friction data (contains coefficient of friction for each grid cell adressed by its index)

        # load friction map file and set up cKDtree for grid representation
        map_coords = np.loadtxt(tpamap_path, comments='#', delimiter=';')
        self.tpa_map = cKDTree(map_coords)

        # load friction data file and set up dictionary with grid cell index as key as mue value as value
        with open(tpadata_path, 'r') as fh:
            tpadata_dict_string = json.load(fh)

        self.tpa_data = {int(k): np.asarray(v) for k, v in tpadata_dict_string.items()}

    def get_friction_singlepos(self,
                               positions: np.ndarray) -> np.array:
        """
        This function returns the friction value mue for a given position.

        Inputs:
        positions:          x,y coordinate(s) in meters from origin for position of requested friction value(s)
                            [[x_0, y_0], [x_1, y_1], ...] (multiple coordinate points allowed)

        Outputs:
        mue_singlepos:      array with coefficient of friction for requested positions (same number)
                            [[mue_0], [mue_1], [mue_2], ...]]
        """

        # check input
        if positions.size == 0:
            return np.asarray([])

        # query requested positions to get indices of grid cells containing the corresponding mue values
        _, idxs = self.tpa_map.query(positions)

        # get mue-value(s) from dictionary
        mue_singlepos = []

        for idx in idxs:
            mue_singlepos.append(self.tpa_data[idx])

        return np.asarray(mue_singlepos)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    inputs_path = os.path.join(module_path, "inputs", "frictionmaps")

    tpamap_path_ = os.path.join(inputs_path, "berlin_2018_tpamap.csv")
    tpadata_path_ = os.path.join(inputs_path, "berlin_2018_tpadata.json")

    mapint = FrictionMapInterface(tpamap_path=tpamap_path_,
                                  tpadata_path=tpadata_path_)

    position_ = np.asarray([[100.0, -80.0],
                            [160.0, 560.0],
                            [133.0, 20.0],
                            [122.0, 10.0],
                            [110.0, 64.0],
                            [131.0, 45.0],
                            [113.0, -58.0],
                            [111.0, -21.0]])

    mue = mapint.get_friction_singlepos(position_)
    print(mue)

    position_ = np.asarray([[0.0, 0.0]])
    _ = mapint.get_friction_singlepos(position_)

    position_ = np.random.rand(300, 2)
    _ = mapint.get_friction_singlepos(position_)

    position_ = np.asarray([])
    _ = mapint.get_friction_singlepos(position_)

    print('INFO: FrictionMapInterface tests passed!')
