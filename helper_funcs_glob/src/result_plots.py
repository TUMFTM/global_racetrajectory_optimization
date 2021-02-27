import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trajectory_planning_helpers


def result_plots(plot_opts: dict,
                 width_veh_opt: float,
                 width_veh_real: float,
                 refline: np.ndarray,
                 bound1_imp: np.ndarray,
                 bound2_imp: np.ndarray,
                 bound1_interp: np.ndarray,
                 bound2_interp: np.ndarray,
                 trajectory: np.ndarray) -> None:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function plots several figures containing relevant trajectory information after trajectory optimization.

    Inputs:
    plot_opts:      dict containing the information which figures to plot
    width_veh_opt:  vehicle width used during optimization in m
    width_veh_real: real vehicle width in m
    refline:        contains the reference line coordinates [x_m, y_m]
    bound1_imp:     first track boundary (as imported) (mostly right) [x_m, y_m]
    bound2_imp:     second track boundary (as imported) (mostly left) [x_m, y_m]
    bound1_interp:  first track boundary (interpolated) (mostly right) [x_m, y_m]
    bound2_interp:  second track boundary (interpolated) (mostly left) [x_m, y_m]
    trajectory:     trajectory data [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    if plot_opts["raceline"]:
        # calculate vehicle boundary points (including safety margin in vehicle width)
        normvec_normalized_opt = trajectory_planning_helpers.calc_normal_vectors.\
            calc_normal_vectors(trajectory[:, 3])

        veh_bound1_virt = trajectory[:, 1:3] + normvec_normalized_opt * width_veh_opt / 2
        veh_bound2_virt = trajectory[:, 1:3] - normvec_normalized_opt * width_veh_opt / 2

        veh_bound1_real = trajectory[:, 1:3] + normvec_normalized_opt * width_veh_real / 2
        veh_bound2_real = trajectory[:, 1:3] - normvec_normalized_opt * width_veh_real / 2

        point1_arrow = refline[0]
        point2_arrow = refline[3]
        vec_arrow = point2_arrow - point1_arrow

        # plot track including optimized path
        plt.figure()
        plt.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7)
        plt.plot(veh_bound1_virt[:, 0], veh_bound1_virt[:, 1], "b", linewidth=0.5)
        plt.plot(veh_bound2_virt[:, 0], veh_bound2_virt[:, 1], "b", linewidth=0.5)
        plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
        plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
        plt.plot(bound1_interp[:, 0], bound1_interp[:, 1], "k-", linewidth=0.7)
        plt.plot(bound2_interp[:, 0], bound2_interp[:, 1], "k-", linewidth=0.7)
        plt.plot(trajectory[:, 1], trajectory[:, 2], "r-", linewidth=0.7)

        if plot_opts["imported_bounds"] and bound1_imp is not None and bound2_imp is not None:
            plt.plot(bound1_imp[:, 0], bound1_imp[:, 1], "y-", linewidth=0.7)
            plt.plot(bound2_imp[:, 0], bound2_imp[:, 1], "y-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1],
                 head_width=7.0, head_length=7.0, fc='g', ec='g')
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.show()

    if plot_opts["raceline_curv"]:
        # plot curvature profile
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 4])
        plt.grid()
        plt.xlabel("distance in m")
        plt.ylabel("curvature in rad/m")
        plt.show()

    if plot_opts["racetraj_vel_3d"]:
        scale_x = 1.0
        scale_y = 1.0
        scale_z = 0.3  # scale z axis such that it does not appear stretched

        # create 3d plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # recast get_proj function to use scaling factors for the axes
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1.0]))

        # plot raceline and boundaries
        ax.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7)
        ax.plot(bound1_interp[:, 0], bound1_interp[:, 1], 0.0, "k-", linewidth=0.7)
        ax.plot(bound2_interp[:, 0], bound2_interp[:, 1], 0.0, "k-", linewidth=0.7)
        ax.plot(trajectory[:, 1], trajectory[:, 2], "r-", linewidth=0.7)

        ax.grid()
        ax.set_aspect("auto")
        ax.set_xlabel("east in m")
        ax.set_ylabel("north in m")

        # plot velocity profile in 3D
        ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 5], color="k")
        ax.set_zlabel("velocity in m/s")

        # plot vertical lines visualizing acceleration and deceleration zones
        ind_stepsize = int(np.round(plot_opts["racetraj_vel_3d_stepsize"] / trajectory[1, 0] - trajectory[0, 0]))
        if ind_stepsize < 1:
            ind_stepsize = 1

        cur_ind = 0
        no_points_traj_vdc = np.shape(trajectory)[0]

        while cur_ind < no_points_traj_vdc - 1:
            x_tmp = [trajectory[cur_ind, 1], trajectory[cur_ind, 1]]
            y_tmp = [trajectory[cur_ind, 2], trajectory[cur_ind, 2]]
            z_tmp = [0.0, trajectory[cur_ind, 5]]  # plot line with height depending on velocity

            # get proper color for line depending on acceleration
            if trajectory[cur_ind, 6] > 0.0:
                col = "g"
            elif trajectory[cur_ind, 6] < 0.0:
                col = "r"
            else:
                col = "gray"

            # plot line
            ax.plot(x_tmp, y_tmp, z_tmp, color=col)

            # increment index
            cur_ind += ind_stepsize

        plt.show()

    if plot_opts["spline_normals"]:
        plt.figure()

        plt.plot(refline[:, 0], refline[:, 1], 'k-')
        for i in range(bound1_interp.shape[0]):
            temp = np.vstack((bound1_interp[i], bound2_interp[i]))
            plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")

        plt.show()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
