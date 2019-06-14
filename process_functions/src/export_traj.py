import numpy as np
import uuid
import hashlib

"""
Created by:
Alexander Heilmeier

Created on:
22.10.2018

Documentation:
This function is used to export the generated trajectory into several files for further usage in the local trajectory
planner.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs..
"""


def export_traj(file_paths: dict,
                traj_race: np.ndarray):

    # create random UUID
    rand_uuid = str(uuid.uuid4())

    # hash GGV file with SHA1
    with open(file_paths["ggv"], 'br') as fh:
        ggv_content = fh.read()
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    # write UUID and GGV hash into file
    with open(file_paths["racetraj_export"], 'w') as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # export race trajectory
    header = "s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2"
    fmt = "%.7f;%.7f;%.7f;%.7f;%.7f;%.7f;%.7f"
    with open(file_paths["racetraj_export"], 'ab') as fh:
        np.savetxt(fh, traj_race, delimiter=";", fmt=fmt, header=header)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
