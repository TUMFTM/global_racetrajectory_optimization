import numpy as np


def import_csv_track(import_path: str) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Created on:
    21.03.2018

    Documentation:
    Read csv file containing track information in the format [x, y, w_tr_right, w_tr_left].

    Inputs:
    import_path:    desired file location

    Outputs:
    reftrack:       imported reference track [x, y, w_tr_right, w_tr_left]
    """

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT DATA ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load data from csv file
    csv_data_temp = np.loadtxt(import_path, delimiter=';')

    # get coords and track widths out of array
    if np.shape(csv_data_temp)[1] == 3:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2] / 2
        w_tr_l = w_tr_r

    elif np.shape(csv_data_temp)[1] == 4:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2]
        w_tr_l = csv_data_temp[:, 3]

    elif np.shape(csv_data_temp)[1] == 5:  # omit z coordinate in this case
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 3]
        w_tr_l = csv_data_temp[:, 4]

    else:
        raise IOError("Track file cannot be read!")

    # create one array
    reftrack = np.column_stack((refline_, w_tr_r, w_tr_l))

    return reftrack


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
