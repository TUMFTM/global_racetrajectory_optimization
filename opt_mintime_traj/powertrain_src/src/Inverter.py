import casadi as ca


class InverterModel:

    __slots__ = ('pars',
                 'temp_inv_n',
                 'temp_inv_s',
                 'temp_inv',
                 'dtemp',
                 'temp_min',
                 'temp_max',
                 'temp_guess',
                 'f_nlp',
                 'f_sol',
                 'p_in_inv',
                 'p_loss_switch',
                 'p_loss_cond',
                 'p_loss_total',
                 'p_loss_total_all_inverters',
                 'r_inv',
                 'p_losses_opt')

    def __init__(self,
                 pwr_pars: dict):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.04.2020

        Documentation: Inverter class for the optimization of global trajectories for electric race cars implemented in
        the CasADi modeling language.

        Inputs:
        pwr_pars: powertrain parameters defined in the initialization file
        """

        self.pars = pwr_pars

        # --------------------------------------------------------------------------------------------------------------
        # Empty inverter states
        # --------------------------------------------------------------------------------------------------------------

        self.temp_inv_n = None
        self.temp_inv_s = None
        self.temp_inv = None
        self.dtemp = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None

        self.f_nlp = None
        self.f_sol = None

        self.p_in_inv = None
        self.p_loss_switch = None
        self.p_loss_cond = None
        self.p_loss_total = None
        self.p_loss_total_all_inverters = None

        self.r_inv = None

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

        self.temp_inv_n = ca.SX.sym('temp_inv_n')
        self.temp_inv_s = self.pars["temp_inv_max"] - 30
        self.temp_inv = self.temp_inv_s * self.temp_inv_n

        # Define limits and initial guess
        self.temp_min = self.pars["T_env"] / self.temp_inv_s
        self.temp_max = self.pars["temp_inv_max"] / self.temp_inv_s
        self.temp_guess = self.pars["T_env"] / self.temp_inv_s

        self.get_thermal_resistance()

    def get_increment(self,
                      sf: ca.SX,
                      temp_cool_mi: ca.SX,
                      temp_cool_12: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes temperature increment of inverter symbolically (sf * dx/dt = dx/ds)

        Inputs:
        sf: transformation factor dt/ds
        temp_cool_mi: cooling fluid temperature of machine-inverter cooling circuit  [°C]
        temp_cool_12: intermediate temperature within motor-inverter cooling circuit (radiator-motor)  [°C]
        """

        self.dtemp = sf * ((self.p_loss_total * 1000 - (self.temp_inv - (temp_cool_mi + temp_cool_12) / 2)
                            / self.r_inv)
                           / (self.pars["C_therm_inv"]))

    def get_loss(self,
                 i_eff: ca.SX,
                 v_dc: ca.SX,
                 p_out_inv: ca.SX = None):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes total power loss of a single inverter and split into loss effects
        (with detailed models) or loss power of a single e-machine using a simple power fit to measured data
        (input -- output power).
        p_out_inv can be left empty in case of detailed loss model usage.

        Inputs:
        i_eff: effective current through one electric machine [A]
        v_dc: terminal voltage battery [V]
        p_out_inv: output power of single inverter [kW]
        """

        if self.pars["simple_loss"]:

            # Input in single inverter [kW] = inverter output + inverter loss
            self.p_in_inv = \
                self.pars["inverter_simple_a"] * p_out_inv ** 2 \
                + self.pars["inverter_simple_b"] * p_out_inv \
                + self.pars["inverter_simple_c"]

            # Total loss [kW]
            self.p_loss_total = (self.p_in_inv - p_out_inv)

        else:

            # Power loss switching [W]
            p_loss_switch = (v_dc / self.pars["V_ref"]) \
                * ((3 * self.pars["f_sw"]) * (i_eff / self.pars["I_ref"])
                   * (self.pars["E_on"] + self.pars["E_off"] + self.pars["E_rr"]))

            # Power loss conducting [W]
            p_loss_cond = 3 * i_eff * (self.pars["V_ce_offset"] + (self.pars["V_ce_slope"] * i_eff))

            # Loss effects [kW]
            self.p_loss_switch = 0.001 * p_loss_switch
            self.p_loss_cond = 0.001 * p_loss_cond

            # Total loss [kW]
            self.p_loss_total = (p_loss_switch + p_loss_cond) * 0.001

    def get_inverters_cum_losses(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculate total loss of all inverters in electric powertrain
        """

        self.p_loss_total_all_inverters = self.p_loss_total * self.pars["N_machines"]

    def get_thermal_resistance(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculates thermal resistance of inverter
        """

        # Thermal resistance inverter [K/W]
        self.r_inv = 1 / (self.pars["h_fluid_mi"] * self.pars["A_cool_inv"])

    def ini_nlp_state(self,
                      x: ca.SX,
                      u: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to define inverter states in NLP

        Inputs:
        x: discrete NLP state
        u: discrete NLP control input
        """

        if self.pars["simple_loss"]:
            self.f_nlp = \
                ca.Function('f_nlp',
                            [x, u], [self.p_loss_total, self.p_in_inv],
                            ['x', 'u'], ['p_loss_total', 'p_inv_in'])
        else:
            self.f_nlp = \
                ca.Function('f_nlp',
                            [x, u], [self.p_loss_total, self.p_loss_switch, self.p_loss_cond],
                            ['x', 'u'], ['p_loss_total', 'p_loss_switch', 'p_loss_cond'])

    def extract_sol(self,
                    w: ca.SX,
                    sol_states: ca.DM):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to retrieve values of optimized NLP inverter

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
            self.p_in_inv = p_losses_opt[1::2]

        else:
            self.f_sol = \
                ca.Function('f_sol',
                            [w], [self.p_losses_opt],
                            ['w'], ['p_losses_opt'])

            # Overwrite lists with optimized numeric values
            p_losses_opt = self.f_sol(sol_states)

            self.p_loss_total = p_losses_opt[0::3]
            self.p_loss_switch = p_losses_opt[1::3]
            self.p_loss_cond = p_losses_opt[2::3]


if __name__ == "__main__":
    pass
