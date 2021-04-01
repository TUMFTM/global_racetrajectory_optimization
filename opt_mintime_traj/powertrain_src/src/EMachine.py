import casadi as ca
import numpy as np


class EMachineModel:

    __slots__ = ('pars',
                 'temp_mot_n',
                 'temp_mot_s',
                 'temp_mot',
                 'dtemp',
                 'temp_min',
                 'temp_max',
                 'temp_guess',
                 'f_nlp',
                 'f_sol',
                 'i_eff',
                 'omega_machine',
                 'p_input',
                 'p_loss_copper',
                 'p_loss_stator_iron',
                 'p_loss_rotor',
                 'p_loss_total',
                 'p_loss_total_all_machines',
                 'r_machine',
                 'p_losses_opt')

    def __init__(self,
                 pwr_pars: dict):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.04.2020

        Documentation: E-Machine class for the optimization of global trajectories for electric race cars implemented in
        the CasADi modeling language.

        Inputs:
        pwr_pars: powertrain parameters defined in the initialization file
        """

        # Store powertrain parameters
        self.pars = pwr_pars

        # --------------------------------------------------------------------------------------------------------------
        # Empty machine states
        # --------------------------------------------------------------------------------------------------------------
        self.temp_mot_n = None
        self.temp_mot_s = None
        self.temp_mot = None
        self.dtemp = None
        self.temp_min = None
        self.temp_max = None
        self.temp_guess = None

        self.f_nlp = None
        self.f_sol = None

        self.i_eff = None
        self.omega_machine = None

        self.p_input = None
        self.p_loss_copper = None
        self.p_loss_stator_iron = None
        self.p_loss_rotor = None
        self.p_loss_total = None
        self.p_loss_total_all_machines = None

        self.r_machine = None

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

        self.temp_mot_n = ca.SX.sym('temp_mot_n')
        self.temp_mot_s = self.pars["temp_mot_max"] - 50
        self.temp_mot = self.temp_mot_s * self.temp_mot_n

        # Define limits and initial guess
        self.temp_min = self.pars["T_env"] / self.temp_mot_s
        self.temp_max = self.pars["temp_mot_max"] / self.temp_mot_s
        self.temp_guess = self.pars["T_env"] / self.temp_mot_s

        self.get_thermal_resistance()

    def get_states(self,
                   f_drive: ca.SX,
                   v: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes states of the electric machine symbolically

        Inputs:
        f_drive: drive force at wheel [N]
        v: velocity of vehicle [m/s]
        """

        # Effective current through single electric machine [A]
        self.i_eff = (f_drive * self.pars["r_wheel"] / self.pars["MotorConstant"] / self.pars["transmission"]) / \
            self.pars["N_machines"]

        # Rotational speed electric machine > speed wheel; v = 2 pi f r [rpm]
        self.omega_machine = v / (2 * np.pi * self.pars["r_wheel"]) * self.pars["transmission"] * 60

    def get_increment(self,
                      sf: ca.SX,
                      temp_cool_12: ca.SX,
                      temp_cool_13: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes temperature increment of electric machine symbolically (sf * dx/dt = dx/ds)

        Inputs:
        sf: transformation factor dt/ds
        temp_cool_12: intermediate temperature within motor-inverter cooling circuit (radiator-motor) [°C]
        temp_cool_13: intermediate temperature within motor-inverter cooling circuit (motor-inverter) [°C]
        """

        self.dtemp = sf * ((self.p_loss_total * 1000 - (self.temp_mot - (temp_cool_12 + temp_cool_13) / 2)
                            / self.r_machine)
                           / (self.pars["C_therm_machine"]))

    def get_loss(self,
                 p_wheel: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Initializes total power loss of a single electric machine and split into loss effects
        (with detailed models) or loss power of a single e-machine using a simple power fit to measured data
        (input -- output power)

        Inputs:
        p_wheel: on wheels desired power [kW]
        """

        if self.pars["simple_loss"]:

            # Input in machine [kW] = P_wheel + machine loss
            p_machine_in = \
                self.pars["machine_simple_a"] * (p_wheel / self.pars["N_machines"]) ** 2 \
                + self.pars["machine_simple_b"] * (p_wheel / self.pars["N_machines"]) \
                + self.pars["machine_simple_c"]

            self.p_input = p_machine_in

            # Machine loss [kW]
            self.p_loss_total = p_machine_in - p_wheel / self.pars["N_machines"]

        else:

            temp_mot = self.temp_mot
            omega_machine = self.omega_machine
            i_eff = self.i_eff

            # Copper loss [W]
            p_loss_copper = \
                (
                 (((temp_mot - 20) * self.pars["C_TempCopper"]) + 1) * self.pars["R_Phase"]
                ) * (i_eff ** 2) * (3 / 2)

            # Stator iron loss [W]
            p_loss_stator_iron = \
                2.885e-13 * omega_machine ** 4 \
                - 1.114e-08 * omega_machine ** 3 \
                + 0.0001123 * omega_machine ** 2 \
                + 0.1657 * omega_machine \
                + 272

            # Rotor loss [W]
            p_loss_rotor = \
                8.143e-14 * omega_machine ** 4 \
                - 2.338e-09 * omega_machine ** 3 \
                + 1.673e-05 * omega_machine ** 2 \
                + 0.112 * omega_machine \
                - 113.6

            # Total loss [kW]
            p_loss_total = (p_loss_copper
                            + p_loss_stator_iron
                            + p_loss_rotor) * 0.001

            # Store losses [kW]
            self.p_loss_copper = 0.001 * p_loss_copper
            self.p_loss_stator_iron = 0.001 * p_loss_stator_iron
            self.p_loss_rotor = 0.001 * p_loss_rotor
            self.p_loss_total = p_loss_total

    def get_machines_cum_losses(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculate total loss of all e-machines in electric powertrain
        """

        self.p_loss_total_all_machines = self.p_loss_total * self.pars["N_machines"]

    def get_thermal_resistance(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Calculates thermal resistance of electric machine to be used within a
        lumped thermal model description
        """

        A_cool_machine = 2 * np.pi * self.pars["r_stator_ext"] * self.pars["l_machine"] * \
            self.pars["A_cool_inflate_machine"]

        # Thermal conduction stator [K/W]
        r_cond_stator = np.log(self.pars["r_stator_ext"] /
                               self.pars["r_stator_int"]) / (2 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])

        # Thermal conduction rotor [K/W]
        r_cond_rotor = np.log(self.pars["r_rotor_ext"] / self.pars["r_rotor_int"]) / \
            (2 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])

        # Thermal conduction shaft [K/W]
        r_cond_shaft = 1 / (4 * np.pi * self.pars["k_iro"] * self.pars["l_machine"])

        # Thermal convection stator -- cooling liquid [K/W]
        r_conv_fluid = 1 / (self.pars["h_fluid_mi"] * A_cool_machine)

        # Thermal resistance by convection in the machine air gap rotor -- stator [K/W]
        r_conv_airgap = 1 / (2 * np.pi * self.pars["h_air_gap"] * self.pars["r_stator_int"] * self.pars["l_machine"])

        # Thermal resistance out [K/W]
        r1 = r_cond_stator + r_conv_fluid
        # Thermal resistance in [K/W]
        r2 = r_cond_shaft + r_cond_rotor + r_conv_airgap

        # Thermal resistance machine [K/W]
        self.r_machine = (r1 * r2) / (r1 + r2)

    def ini_nlp_state(self,
                      x: ca.SX,
                      u: ca.SX):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to define e-machine states in NLP

        Inputs:
        x: discrete NLP state
        u: discrete NLP control input
        """

        if self.pars["simple_loss"]:
            self.f_nlp = \
                ca.Function('f_nlp',
                            [x, u], [self.p_loss_total, self.p_input],
                            ['x', 'u'], ['p_loss_total', 'p_input'])
        else:
            self.f_nlp = \
                ca.Function('f_nlp',
                            [x, u], [self.p_loss_total, self.p_loss_copper, self.p_loss_stator_iron, self.p_loss_rotor,
                                     self.i_eff],
                            ['x', 'u'], ['p_loss_total', 'p_loss_copper', 'p_loss_stator_iron', 'p_loss_rotor',
                                         'i_eff'])

    def extract_sol(self,
                    w: ca.SX,
                    sol_states: ca.DM):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.04.2020

        Documentation: Defines function to retrieve values of optimized NLP e-machine

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
            self.p_input = p_losses_opt[1::2]

        else:
            self.f_sol = \
                ca.Function('f_sol',
                            [w], [self.p_losses_opt],
                            ['w'], ['p_losses_opt'])

            # Overwrite lists with optimized numeric values
            p_losses_opt = self.f_sol(sol_states)

            self.p_loss_total = p_losses_opt[0::5]
            self.p_loss_copper = p_losses_opt[1::5]
            self.p_loss_stator_iron = p_losses_opt[2::5]
            self.p_loss_rotor = p_losses_opt[3::5]
            self.i_eff = p_losses_opt[4::5]


if __name__ == "__main__":
    pass
