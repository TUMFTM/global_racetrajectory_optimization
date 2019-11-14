import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trajectory_planning_helpers

"""
Created by:
Alexander Heilmeier

Documentation:
This function plots several diagrams after trajectory optimization.

The process_functions folder only contains functions that are used to outsource code snippets from the main script.
Therefore, these functions cannot be seen as independent functional units and their inputs and outputs are undocumented.
Please have a look into the main_globaltraj.py script to understand inputs and outputs.
"""


def plot_funcs(plot_opts: dict,
               optim_opts: dict,
               veh_dims: dict,
               refline_interp: np.ndarray,
               bound1: np.ndarray,
               bound2: np.ndarray,
               trajectory_opt: np.ndarray):

    if plot_opts["raceline"]:
        # calc vehicle boundary points (including safety margin in vehicle width)
        normvec_normalized_opt = trajectory_planning_helpers.calc_normal_vectors.\
            calc_normal_vectors(trajectory_opt[:, 3])

        raceline_interp = trajectory_opt[:, 1:3]

        veh_bound1_virt = raceline_interp + normvec_normalized_opt * optim_opts["w_veh"] / 2
        veh_bound2_virt = raceline_interp - normvec_normalized_opt * optim_opts["w_veh"] / 2

        veh_bound1_real = raceline_interp + normvec_normalized_opt * veh_dims["w_veh_real"] / 2
        veh_bound2_real = raceline_interp - normvec_normalized_opt * veh_dims["w_veh_real"] / 2

        point1_arrow = refline_interp[0]
        point2_arrow = refline_interp[3]
        vec_arrow = point2_arrow - point1_arrow

        # plot track including optimized path
        plt.figure()
        plt.plot(refline_interp[:, 0], refline_interp[:, 1], "k--", linewidth=0.7)
        plt.plot(veh_bound1_virt[:, 0], veh_bound1_virt[:, 1], "b", linewidth=0.5)
        plt.plot(veh_bound2_virt[:, 0], veh_bound2_virt[:, 1], "b", linewidth=0.5)
        plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
        plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
        plt.plot(bound1[:, 0], bound1[:, 1], "k-", linewidth=0.7)
        plt.plot(bound2[:, 0], bound2[:, 1], "k-", linewidth=0.7)
        plt.plot(trajectory_opt[:, 1], trajectory_opt[:, 2], "r-", linewidth=0.7)
        plt.grid()
        ax = plt.gca()
        ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1], head_width=7.0, head_length=7.0,
                 fc='g', ec='g')
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.show()

    if plot_opts["curv_profile"]:
        # plot curvature profile
        plt.figure()
        plt.plot(trajectory_opt[:, 0], trajectory_opt[:, 4])
        plt.grid()
        plt.xlabel("distance in m")
        plt.ylabel("curvature in radpm")
        plt.legend(["kappa after opt"])
        plt.show()

    if plot_opts["velprofile_3d"]:
        scale_x = 1.0
        scale_y = 1.0
        scale_z = 0.3  # scale z axis such that it does not appear stretched

        # create 3d plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # recast get_proj function to use scaling factors for the axes
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1.0]))

        # plot raceline and boundaries
        ax.plot(refline_interp[:, 0], refline_interp[:, 1], "k--", linewidth=0.7)
        ax.plot(bound1[:, 0], bound1[:, 1], 0.0, "k-", linewidth=0.7)
        ax.plot(bound2[:, 0], bound2[:, 1], 0.0, "k-", linewidth=0.7)
        ax.plot(trajectory_opt[:, 1], trajectory_opt[:, 2], "r-", linewidth=0.7)

        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel("east in m")
        ax.set_ylabel("north in m")

        # plot velocity profile in 3D
        ax.plot(trajectory_opt[:, 1], trajectory_opt[:, 2], trajectory_opt[:, 5], color="k")
        ax.set_zlabel("velocity in m/s")

        # plot vertical lines visualizing acceleration and deceleration zones
        ind_stepsize = int(np.round(plot_opts["velprofile_3d_stepsize"] / trajectory_opt[1, 0] - trajectory_opt[0, 0]))
        if ind_stepsize < 1:
            ind_stepsize = 1

        cur_ind = 0
        no_points_traj_vdc = np.shape(trajectory_opt)[0]

        while cur_ind < no_points_traj_vdc - 1:
            x_tmp = [trajectory_opt[cur_ind, 1], trajectory_opt[cur_ind, 1]]
            y_tmp = [trajectory_opt[cur_ind, 2], trajectory_opt[cur_ind, 2]]
            z_tmp = [0.0, trajectory_opt[cur_ind, 5]]  # plot line with height depending on velocity

            # get proper color for line depending on acceleration
            if trajectory_opt[cur_ind, 6] > 0.0:
                col = "g"
            elif trajectory_opt[cur_ind, 6] < 0.0:
                col = "r"
            else:
                col = "gray"

            # plot line
            ax.plot(x_tmp, y_tmp, z_tmp, color=col)

            # increment index
            cur_ind += ind_stepsize

        plt.show()

    if plot_opts["spline_normals"]:
        # plot normals
        plt.figure()
        for i in range(bound1.shape[0]):
            temp = np.vstack((bound1[i], bound2[i]))
            plt.plot(temp[:, 0], temp[:, 1], "k-", linewidth=0.7)
        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.show()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
