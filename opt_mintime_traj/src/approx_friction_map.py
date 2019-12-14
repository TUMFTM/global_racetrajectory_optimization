import numpy as np
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import opt_mintime_traj


def approx_friction_map(reftrack: np.ndarray,
                        normvectors: np.ndarray,
                        tpamap_path: str,
                        tpadata_path: str,
                        pars: dict,
                        dn: float,
                        n_gauss: int,
                        print_debug: bool,
                        plot_debug: bool) -> tuple:
    """
    Created by:
    Fabian Christ

    Documentation:
    A simplified dependency between the friction coefficients (mue) and the lateral distance to the reference line (n)
    is obtained for each wheel along the racetrack. For this purpose friction coefficients are determined for a fine 
    grid on the normal vectors from the friction map. Then the dependency between the extracted friction coefficients 
    and the decision variable n for each path coordinate s_k is described by linear equations (var_friction: "lienar") 
    or by linear regression with gaussian basis functions (var_friction: "gauss").

    Inputs:
    reftrack:       track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvectors:    array containing normalized normal vectors for every traj. point [x_component, y_component]
    tpamap_path:    file path to tpa map (required for friction map loading)
    tpadata_path:   file path to tpa data (required for friction map loading)
    pars:           parameters dictionary
    dn:             distance of equidistant points on normal vectors for extracting the friction coefficients
    n_gauss:        number of gaussian basis functions on each side (n_gauss_tot = 2 * n_gauss + 1)
    print_debug:    determines if debug prints are shown
    plot_debug:     determines if debug plots are shown

    Outputs:
    w_mue_fl:       parameters for friction map approximation along the racetrack (left front wheel)
    w_mue_fr:       parameters for friction map approximation along the racetrack (right front wheel)
    w_mue_rl:       parameters for friction map approximation along the racetrack (left rear wheel)
    w_mue_rr:       parameters for friction map approximation along the racetrack (right rear wheel)
    center_dist     distance between two gaussian basis functions along the racetrack (only for var_friction: "gauss")
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATION ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # extract friction coefficients from friction map
    n, mue_fl, mue_fr, mue_rl, mue_rr = opt_mintime_traj.src.extract_friction_coeffs.\
        extract_friction_coeffs(reftrack=reftrack,
                                normvectors=normvectors,
                                tpamap_path=tpamap_path,
                                tpadata_path=tpadata_path,
                                pars=pars,
                                dn=dn,
                                print_debug=print_debug,
                                plot_debug=plot_debug)

    # number of steps along the reference line
    num_steps = len(n)

    # number of guassian basis functions
    n_gauss_tot = 2 * n_gauss + 1

    # initialize solution vectors
    if pars["optim_opts"]["var_friction"] == 'linear':
        w_mue_fl = np.zeros((num_steps, 2))
        w_mue_fr = np.zeros((num_steps, 2))
        w_mue_rl = np.zeros((num_steps, 2))
        w_mue_rr = np.zeros((num_steps, 2))
        center_dist = np.zeros((num_steps, 1))

    elif pars["optim_opts"]["var_friction"] == "gauss":
        w_mue_fl = np.zeros((num_steps, n_gauss_tot + 1))
        w_mue_fr = np.zeros((num_steps, n_gauss_tot + 1))
        w_mue_rl = np.zeros((num_steps, n_gauss_tot + 1))
        w_mue_rr = np.zeros((num_steps, n_gauss_tot + 1))
        center_dist = np.zeros((num_steps, 1))

    else:
        raise ValueError('Unknown method for friction map approximation!')

    if plot_debug:
        plt.figure(1)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATION ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    for i in range(num_steps):
        if pars["optim_opts"]["var_friction"] == "linear":
            w_mue_fl[i, :] = np.polyfit(n[i], mue_fl[i].T[0], 1)
            w_mue_fr[i, :] = np.polyfit(n[i], mue_fr[i].T[0], 1)
            w_mue_rl[i, :] = np.polyfit(n[i], mue_rl[i].T[0], 1)
            w_mue_rr[i, :] = np.polyfit(n[i], mue_rr[i].T[0], 1)

        elif pars["optim_opts"]["var_friction"] == "gauss":
            # get distance between center of gaussian basis functions
            center_dist[i, 0] = (n[i][-1] - n[i][0]) / (2 * n_gauss)

            # regression with gaussian basis functions (see class definition below)
            gauss_model = make_pipeline(GaussianFeatures(n_gauss_tot), LinearRegression())

            # regression for front left wheel
            gauss_model.fit(n[i][:, np.newaxis], mue_fl[i])
            w_mue_fl[i, :n_gauss_tot] = gauss_model._final_estimator.coef_[0]
            w_mue_fl[i, n_gauss_tot] = gauss_model._final_estimator.intercept_[0]

            # regression for front right wheel
            gauss_model.fit(n[i][:, np.newaxis], mue_fr[i])
            w_mue_fr[i, :n_gauss_tot] = gauss_model._final_estimator.coef_[0]
            w_mue_fr[i, n_gauss_tot] = gauss_model._final_estimator.intercept_[0]

            # regression for rear left wheel
            gauss_model.fit(n[i][:, np.newaxis], mue_rl[i])
            w_mue_rl[i, :n_gauss_tot] = gauss_model._final_estimator.coef_[0]
            w_mue_rl[i, n_gauss_tot] = gauss_model._final_estimator.intercept_[0]

            # regression for rear right wheel
            gauss_model.fit(n[i][:, np.newaxis], mue_rr[i])
            w_mue_rr[i, :n_gauss_tot] = gauss_model._final_estimator.coef_[0]
            w_mue_rr[i, n_gauss_tot] = gauss_model._final_estimator.intercept_[0]

        if print_debug:
            tph.progressbar.progressbar(i, num_steps, 'Approximation of friction map')

        if plot_debug and pars["optim_opts"]["var_friction"] == "linear":
            n_fit = np.linspace(n[i][0], n[i][-1], 3)
            plt.scatter(n[i], mue_rr[i])
            plt.plot(n_fit, w_mue_rr[i, 0] * n_fit + w_mue_rr[i, 1])

        elif plot_debug and pars["optim_opts"]["var_friction"] == "gauss":
            n_fit = np.linspace(n[i][0], n[i][-1], 100)
            plt.scatter(n[i], mue_rr[i])
            plt.plot(n_fit, gauss_model.predict(n_fit[:, np.newaxis]))

    if plot_debug:
        plt.xlabel('n in m')
        plt.ylabel(r'$\it{\mu}$')
        plt.title('Approximation of friction map (e.g. for tire rear right)')
        plt.show()

    return w_mue_fl, w_mue_fr, w_mue_rl, w_mue_rr, center_dist


# ----------------------------------------------------------------------------------------------------------------------
# GAUSSIAN FEATURES CLASS ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# uniformly spaced Gaussian features for one-dimensional input
class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
        self.centers_ = None
        self.width_ = None

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
