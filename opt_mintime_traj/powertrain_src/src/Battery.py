import casadi as ca
import numpy as np


class BattModel:

    __slots__ = ('pars',
                 'temp_batt_n',
                 'temp_batt_s',
                 'temp_batt',
                 'dtemp',
                 'dsoc',
                 'temp_min',
                 'temp_max',
                 'temp_guess',
                 'soc_min',
                 'soc_max',
                 'soc_guess',
                 'soc_batt_n',
                 'soc_batt_s',
                 'soc_batt',
                 'v_dc',
                 'i_batt',
                 'f_nlp',
                 'f_sol',
                 'p_loss_total',
                 'p_out_batt',
                 'p_internal_batt',
                 'r_batt_inverse',
                 'p_losses_opt')

    def __init__(self,
                 pwr_pars: dict):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.04.2020

        Documentation: Battery class for the optimization of global trajectories for electric race cars implemented in
        the CasADi modeling language.

        Inputs:
        pwr_pars: powertrain parameters defined in the initialization file
        """

        # Store powertrain parameters
        self.pars = pwr_pars

        # --------------------------------------------------------------------------------------------------------------
        # Empty battery states
        # --------------------------------------------------------------------------------------------------------------
        self.temp_batt_n = None
        self.temp_batt_s = None
        self.temp_batt = None
        self.dtemp = None
        self.dsoc = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None
        self.soc_min = None
        self.soc_max = None
        self.soc_guess = None
        self.soc_batt_n = None
        self.soc_batt_s = None
        self.soc_batt = None

        self.v_dc = None
        self.i_batt = None

        self.f_nlp = None
        self.f_sol = None

        self.p_loss_total = None
        self.p_out_batt = None
        self.p_internal_batt = None

        self.r_batt_inverse = None

        # Optimized losses list: p_loss_total, p_loss_effects
        self.p_losses_opt = []

        # Call initialization function
        self.initialize()

    def initialize(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initialization of necessary optimization variables (symbolic CasADi expressions)
        and states including limits.
        """

        # battery temperature [°C]
        self.temp_batt_n = ca.SX.sym('temp_batt_n')
        self.temp_batt_s = self.pars["temp_batt_max"] - 10
        self.temp_batt = self.temp_batt_s * self.temp_batt_n

        # Define limits and initial guess
        self.temp_min = self.pars["T_env"] / self.temp_batt_s
        self.temp_max = self.pars["temp_batt_max"] / self.temp_batt_s
        self.temp_guess = self.pars["T_env"] / self.temp_batt_s

        # SOC of battery [-]
        self.soc_batt_n = ca.SX.sym('soc_batt_n')
        self.soc_batt_s = 1
        self.soc_batt = self.soc_batt_s * self.soc_batt_n

        self.soc_min = 0 / self.soc_batt_s
        self.soc_max = 1 / self.soc_batt_s
        self.soc_guess = 0.5

        self.get_thermal_resistance()

        self.ocv_voltage()

    def get_increment(self,
                      sf: ca.SX,
                      temp_cool_b: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes symbolic temperature increment of battery (sf * dx/dt = dx/ds)

        Inputs:
        sf: transformation factor dt/ds
        temp_cool_b: battery cooling liquid temperature [°C]
        """

        self.dtemp = \
            sf * ((self.p_loss_total * 1000 - self.r_batt_inverse * (self.temp_batt - temp_cool_b)) /
                  (self.pars["C_therm_cell"] * self.pars["N_cells_serial"] * self.pars["N_cells_parallel"]))

    def get_soc(self,
                sf: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initialize SOC increment of battery (sf * dx/dt = dx/ds)
        """

        self.dsoc = - sf * ((self.p_out_batt + self.p_loss_total) / 3600 / self.pars["C_batt"])

    def battery_loss(self,
                     p_des: ca.SX,
                     p_loss_inv: ca.SX,
                     p_loss_mot: ca.SX,
                     p_in_inv: ca.SX = None):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines a battery loss model that was trained on measurement data based on Gaussian Processes

        Inputs:
        p_des: on wheels desired power [kW]
        p_loss_inv: inverter losses of all the inverters in the electric powertrain [kW]
        p_loss_mot: machine losses of all the electric machine in the electric powertrain [kW]
        p_in_inv: input power into a single inverter in the electric powertrain [kW]
        """

        if self.pars["simple_loss"]:

            p_in_inv *= self.pars["N_machines"]

            # Calculation of battery internal power using simple OCV model with one resistance [W]
            p_internal_batt = ((self.pars["V_OC_simple"] ** 2) / (2 * self.pars["R_i_simple"])) - \
                self.pars["V_OC_simple"] \
                * np.sqrt(
                (self.pars["V_OC_simple"] ** 2 - 4 * p_in_inv * 1000 * self.pars["R_i_simple"])) / \
                (2 * self.pars["R_i_simple"])

            # Transform to [kW]
            self.p_internal_batt = 0.001 * p_internal_batt

            # Battery loss [kW]
            self.p_loss_total = self.p_internal_batt - p_in_inv

            # Battery output [kW] = input in all inverters in the powertrain
            self.p_out_batt = p_in_inv

        else:

            print('\033[91m' + 'ERROR: Chosen powertrain loss option unknown!' + '\033[0m')
            exit(1)

    def ocv_voltage(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Outputs the battery Open Circuit Voltage as a 3rd order polynom
        """

        # OCV on battery terminals [V]
        self.v_dc = self.pars["N_cells_serial"] * \
            (1.245 * self.soc_batt ** 3 - 1.679 * self.soc_batt ** 2 + 1.064 * self.soc_batt + 3.566)

    def get_thermal_resistance(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculates thermal resistance of battery to be used within a lumped description
        """

        # Thermal resistance inverse [K/W]
        self.r_batt_inverse = 1 / 0.002

    def ini_nlp_state(self,
                      x: ca.SX,
                      u: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to define battery states in NLP

        Inputs:
        x: discrete NLP state
        u: discrete NLP control input
        """

        if self.pars["simple_loss"]:
            self.f_nlp = \
                ca.Function('f_nlp',
                            [x, u], [self.p_loss_total, self.p_out_batt],
                            ['x', 'u'], ['p_loss_total', 'p_out_batt'])
        else:
            print('\033[91m' + 'ERROR: Chosen powertrain loss option unknown!' + '\033[0m')
            exit(1)

    def extract_sol(self,
                    w: ca.SX,
                    sol_states: ca.DM):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to retrieve values of optimized NLP battery states

        Inputs:
        w: discrete optimized NLP decision variables (x and u)
        sol_states: numeric values belonging to the symbolic NLP decision variables w
        """

        if self.pars["simple_loss"]:
            self.f_sol = \
                ca.Function('f_sol',
                            [w], [self.p_losses_opt],
                            ['w'], ['p_losses_opt'])

            # Overwrite lists with optimized numeric values
            p_losses_opt = self.f_sol(sol_states)

            self.p_loss_total = p_losses_opt[0::2]
            self.p_out_batt = p_losses_opt[1::2]

        else:
            print('\033[91m' + 'ERROR: Chosen powertrain loss option unknown!' + '\033[0m')
            exit(1)


if __name__ == "__main__":
    pass
