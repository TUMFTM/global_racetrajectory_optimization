import numpy as np
import int_globalenv_trajgen
import trajectory_planning_helpers


"""
Created by:
Alexander Heilmeier

Created on:
22.10.2018

Documentation:
This function includes the algorithm part connected to the import of the ggv diagram.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def imp_track_ggv(imp_opts: dict,
                  file_paths: dict,
                  veh_dims: dict) -> tuple:

    # import CSV track (refline + track widths in one array)
    reftrack_imp = int_globalenv_trajgen.src.import_csv_track.import_csv_track(import_path=file_paths["track"])

    # check if imported centerline should be flipped, i.e. reverse direction
    if imp_opts["flip_imp_track"]:
        reftrack_imp = np.flipud(reftrack_imp)

    # check if imported centerline should be reorder for a new starting point
    if imp_opts["set_new_start"]:
        ind_start = np.argmin(np.power(reftrack_imp[:, 0] - imp_opts["new_start"][0], 2)
                              + np.power(reftrack_imp[:, 1] - imp_opts["new_start"][1], 2))
        reftrack_imp = np.roll(reftrack_imp, reftrack_imp.shape[0] - ind_start, axis=0)

    # check minimum track width for vehicle width plus a small safety margin
    w_tr_min = np.amin(reftrack_imp[:, 2] + reftrack_imp[:, 3])

    if w_tr_min < veh_dims["w_veh_real"] + 0.5:
        print("Warning: Minimum track width %.2f m is close to or smaller than vehicle width!" % np.amin(w_tr_min))

    # import ggv diagram
    ggv = trajectory_planning_helpers.import_ggv.import_ggv(ggv_import_path=file_paths["ggv"])

    return reftrack_imp, ggv


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
