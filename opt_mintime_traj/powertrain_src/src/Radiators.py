import casadi as ca


class RadiatorModel:

    __slots__ = ('pars',
                 'temp_cool_mi_n',
                 'temp_cool_mi_s',
                 'temp_cool_mi',
                 'temp_cool_b_n',
                 'temp_cool_b_s',
                 'temp_cool_b',
                 'temp_cool_mi_min',
                 'temp_cool_mi_max',
                 'temp_cool_mi_guess',
                 'temp_cool_b_min',
                 'temp_cool_b_max',
                 'temp_cool_b_guess',
                 'temp_cool_12',
                 'temp_cool_13',
                 'dtemp_cool_mi',
                 'dtemp_cool_b',
                 'r_rad',
                 'f_nlp',
                 'f_sol',
                 'temps_opt')

    def __init__(self,
                 pwr_pars: dict):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.04.2020

        Documentation: Radiators class for the optimization of global trajectories for electric race cars implemented in
        the CasADi modeling language.

        Inputs:
        pwr_pars: powertrain parameters defined in the initialization file
        """

        # Store powertrain parameters
        self.pars = pwr_pars

        # --------------------------------------------------------------------------------------------------------------
        # Empty radiator states
        # --------------------------------------------------------------------------------------------------------------
        self.temp_cool_mi_n = None
        self.temp_cool_mi_s = None
        self.temp_cool_mi = None

        self.temp_cool_b_n = None
        self.temp_cool_b_s = None
        self.temp_cool_b = None

        self.temp_cool_mi_min = None
        self.temp_cool_mi_max = None
        self.temp_cool_mi_guess = None
        self.temp_cool_b_min = None
        self.temp_cool_b_max = None
        self.temp_cool_b_guess = None

        self.temp_cool_12 = None
        self.temp_cool_13 = None

        self.dtemp_cool_mi = None
        self.dtemp_cool_b = None

        self.r_rad = None

        self.f_nlp = None
        self.f_sol = None

        # Optimized temperatures list
        self.temps_opt = []

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

        # cooling liquid temperature for motor and inverter circuit [°C]
        self.temp_cool_mi_n = ca.SX.sym('temp_cool_mi_n')
        self.temp_cool_mi_s = self.pars["temp_inv_max"] - 30
        self.temp_cool_mi = self.temp_cool_mi_s * self.temp_cool_mi_n

        # cooling liquid temperature for battery circuit [°C]
        self.temp_cool_b_n = ca.SX.sym('temp_cool_b_n')
        self.temp_cool_b_s = self.pars["temp_batt_max"] - 10
        self.temp_cool_b = self.temp_cool_b_s * self.temp_cool_b_n

        self.temp_cool_mi_min = self.pars["T_env"] / self.temp_cool_mi_s
        self.temp_cool_mi_max = (self.pars["temp_inv_max"] - 10) / self.temp_cool_mi_s
        self.temp_cool_mi_guess = (self.pars["T_env"]) / self.temp_cool_mi_s

        self.temp_cool_b_min = self.pars["T_env"] / self.temp_cool_b_s
        self.temp_cool_b_max = self.pars["temp_batt_max"] / self.temp_cool_b_s
        self.temp_cool_b_guess = self.pars["T_env"] / self.temp_cool_b_s

        self.get_thermal_resistance()

    def get_thermal_resistance(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculates thermal resistance of single radiator to be used within a
        lumped thermal model description
        """

        self.r_rad = 1 / (self.pars["h_air"] * self.pars["A_cool_rad"])

    def get_intermediate_temps(self,
                               temp_inv: ca.SX,
                               r_inv: float):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Returns intermediate temps motor inverter necessary for thermodynamical modelling
        (motor + inverter circuit)

        Inputs:
        temp_inv: inverter temperature [°C]
        r_inv: inverter thermal resistance [K/W]
        """

        self.temp_cool_12 = \
            (self.temp_cool_mi * (self.pars["c_heat_fluid"] * self.pars["flow_rate_inv"] * r_inv - 1)
             + 2 * temp_inv) / \
            (1 + self.pars["c_heat_fluid"] * self.pars["flow_rate_inv"] * r_inv)

        self.temp_cool_13 = \
            (self.temp_cool_mi * (2 * self.pars["c_heat_fluid"] * self.pars["flow_rate_rad"] * self.r_rad + 1) -
             2 * self.pars["T_env"]) / \
            (-1 + 2 * self.pars["c_heat_fluid"] * self.pars["flow_rate_rad"] * self.r_rad)

    def get_increment_mi(self,
                         sf: ca.SX,
                         temp_mot: ca.SX,
                         temp_inv: ca.SX,
                         r_inv: float,
                         r_machine: float):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes temperature increment of radiator machine-inverter circuit symbolically
        (sf * dx/dt = dx/ds)

        Inputs:
        sf: transformation factor dt/ds
        temp_mot: temeprature of electric machine [°C]
        temp_inv: temeprature of inverter [°C]
        r_inv: thermal resistance of inverter [K/W]
        r_machine: thermal resistance of electric machine [K/W]
        """

        self.dtemp_cool_mi = \
            sf * ((self.pars["N_machines"] * ((temp_mot - (self.temp_cool_12 + self.temp_cool_13) / 2) / r_machine +
                                              (temp_inv - (self.temp_cool_mi + self.temp_cool_12) / 2) / r_inv) -
                   ((self.temp_cool_mi + self.temp_cool_13) / 2 - self.pars["T_env"]) / self.r_rad) /
                  (self.pars["m_therm_fluid_mi"] * self.pars["c_heat_fluid"]))

    def get_increment_b(self,
                        sf: ca.SX,
                        temp_batt: ca.SX,
                        temp_cool_b: ca.SX,
                        R_eq_B_inv: float):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes temperature increment of radiator in battery circuit symbolically
        (sf * dx/dt = dx/ds)

        Inputs:
        sf: transformation factor dt/ds
        temp_batt: temeprature of battery [°C]
        temp_cool_b: temeprature of cooling liquid battery [°C]
        R_eq_B_inv: inverse of thermal resistance of battery [W/K]
        """

        self.dtemp_cool_b = \
            sf * ((R_eq_B_inv * (temp_batt - temp_cool_b) -
                   (temp_cool_b - self.pars["T_env"]) / self.r_rad) /
                  (self.pars["m_therm_fluid_b"] * self.pars["c_heat_fluid"]))

    def ini_nlp_state(self,
                      x: ca.SX,
                      u: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to define radiators' states in NLP

        Inputs:
        x: discrete NLP state
        u: discrete NLP control input
        """

        self.f_nlp = \
            ca.Function('f_nlp',
                        [x, u], [self.temp_cool_mi, self.temp_cool_b],
                        ['x', 'u'], ['temp_cool_mi', 'temp_cool_b'])

    def extract_sol(self,
                    w: ca.SX,
                    sol_states: ca.DM):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to retrieve values of optimized NLP radiators

        Inputs:
        w: discrete optimized NLP decision variables (x and u)
        sol_states: numeric values belonging to the symbolic NLP decision variables w
        """

        self.f_sol = \
            ca.Function('f_sol',
                        [w], [self.temps_opt],
                        ['w'], ['temps_opt'])

        # Overwrite lists with optimized numeric values
        temps_opt = self.f_sol(sol_states)

        self.temp_cool_mi = temps_opt[0::2]
        self.temp_cool_b = temps_opt[1::2]


if __name__ == "__main__":
    pass
