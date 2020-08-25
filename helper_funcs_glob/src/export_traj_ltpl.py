import numpy as np
import uuid
import hashlib


def export_traj_ltpl(file_paths: dict,
                     spline_lengths_opt,
                     trajectory_opt,
                     reftrack,
                     normvec_normalized,
                     alpha_opt) -> None:
    """
    Created by:
    Tim Stahl
    Alexander Heilmeier

    Documentation:
    This function is used to export the generated trajectory into a file for further usage in the local trajectory
    planner on the car (including map information via normal vectors and bound widths). The generated files get an
    unique UUID and a hash of the ggv diagram to be able to check it later.

    The stored trajectory has the following columns:
    [x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m, psi_racetraj_rad,
     kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2]

    Inputs:
    file_paths:         paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    spline_lengths_opt: lengths of the splines on the raceline in m
    trajectory_opt:     generated race trajectory
    reftrack:           track definition [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized: normalized normal vectors on the reference line [x_m, y_m]
    alpha_opt:          solution vector of the opt. problem containing the lateral shift in m for every ref-point
    """

    # convert trajectory to desired format
    s_raceline_preinterp_cl = np.cumsum(spline_lengths_opt)
    s_raceline_preinterp_cl = np.insert(s_raceline_preinterp_cl, 0, 0.0)

    psi_normvec = []
    kappa_normvec = []
    vx_normvec = []
    ax_normvec = []

    for s in list(s_raceline_preinterp_cl[:-1]):
        # get closest point on trajectory_opt
        idx = (np.abs(trajectory_opt[:, 0] - s)).argmin()

        # get data at this index and append
        psi_normvec.append(trajectory_opt[idx, 3])
        kappa_normvec.append(trajectory_opt[idx, 4])
        vx_normvec.append(trajectory_opt[idx, 5])
        ax_normvec.append(trajectory_opt[idx, 6])

    traj_ltpl = np.column_stack((reftrack,
                                 normvec_normalized,
                                 alpha_opt,
                                 s_raceline_preinterp_cl[:-1],
                                 psi_normvec,
                                 kappa_normvec,
                                 vx_normvec,
                                 ax_normvec))
    traj_ltpl_cl = np.vstack((traj_ltpl, traj_ltpl[0]))
    traj_ltpl_cl[-1, 7] = s_raceline_preinterp_cl[-1]

    # create random UUID
    rand_uuid = str(uuid.uuid4())

    # hash ggv file with SHA1
    if "ggv_file" in file_paths:
        with open(file_paths["ggv_file"], 'br') as fh:
            ggv_content = fh.read()
    else:
        ggv_content = np.array([])
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    # write UUID and GGV hash into file
    with open(file_paths["traj_ltpl_export"], 'w') as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # export trajectory data for local planner
    header = "x_ref_m; y_ref_m; width_right_m; width_left_m; x_normvec_m; y_normvec_m; " \
             "alpha_m; s_racetraj_m; psi_racetraj_rad; kappa_racetraj_radpm; vx_racetraj_mps; ax_racetraj_mps2"
    fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
    with open(file_paths["traj_ltpl_export"], 'ab') as fh:
        np.savetxt(fh, traj_ltpl, fmt=fmt, header=header)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
