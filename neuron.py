import numpy as np
from numpy.random import default_rng
from scipy.integrate import solve_ivp


class LIFNeuron:

    # Neuron physiological parameters
    Decay_time = 20.
    Ref_period = 10.
    Firing_threshold = -45. - 1e-10
    Resting_potential = -65.
    Resistance = 10.

    def __init__(self, *, neuron_index: int, column_index: int, type_: int, initial_condition: float, time_step: float,
                 current: float) -> None:
        self.n_idx = neuron_index
        self.col_idx = column_index
        self.fire_times = list()  # all the neuron firing times are stored in a list for future analysis
        self.type = type_  # excitatory (1) or inhibitory (-1)
        self.on = True  # if the neuron is ready to fire, self.on = True. After firing self.on = False for ref_period
        self.off_till = -1  # timer for the neuron off state
        self.v = initial_condition  # initial condition
        # all incoming inputs are stored in a dictionary of format "timestamp: sum of inputs(timestamp)"
        self.incoming_inputs = dict()
        self.tau = int(LIFNeuron.Ref_period / time_step)  # refractory period in integer time steps
        self.decay = np.exp(-time_step / LIFNeuron.Decay_time)  # potential decay on every time step
        # resting potential in the presence of external currents
        self.new_rest = LIFNeuron.Resting_potential + LIFNeuron.Resistance * current

    # adjusts the external current source
    def adjust_current(self, current: float):
        self.new_rest = LIFNeuron.Resting_potential + LIFNeuron.Resistance * current

    # receives inputs from other neurons
    def receive_inputs(self, input_time: int, magnitude: float):
        if input_time < self.off_till:
            # if the neuron will still be off when the input comes, ignore it
            pass
        elif input_time not in self.incoming_inputs:
            # create a "timestamp: input(timestamp)" pair
            self.incoming_inputs[input_time] = magnitude
        else:
            # add input to existing "timestamp"
            self.incoming_inputs[input_time] += magnitude

    # integrates the neuron state variable to the next time step, executes neuron firing
    def propagate_step(self, *, current_time: int, dt: float) -> (float, bool):
        # first check if the neuron is turned on
        if self.on:
            # add inputs and integrate dynamics
            if current_time in self.incoming_inputs:
                self.v += self.incoming_inputs[current_time]
                # wipe inputs for current time so the dictionary doesn't become cluttered
                del self.incoming_inputs[current_time]

            # if firing threshold exceeded, record spike
            if self.v > LIFNeuron.Firing_threshold:
                self.fire_times.append(current_time)

                # turn neuron off, record time until the neuron is turned back on
                self.on = False
                self.off_till = current_time + self.tau
                # off timer randomly varies by a few time steps to avoid a forced period tau in the system
                self.off_till += default_rng().integers(-5, 6, 1)[0]

                return self.v, True

            # propagate the dynamics
            self.v = self.v * self.decay + self.new_rest * (1 - self.decay)

        else:
            # reset membrane potential
            if self.v != LIFNeuron.Resting_potential:
                self.v = LIFNeuron.Resting_potential
            # if refractory period is over, turn the neuron back on
            if current_time >= self.off_till:
                self.on = True

        # wipe inputs for current time so the dictionary doesn't become cluttered
        if current_time in self.incoming_inputs:
            del self.incoming_inputs[current_time]

        return self.v, False


class QIFNeuron:
    # Neuron physiological parameters
    Decay_time = 20.
    Ref_period = 10.
    Resting_potential = -65.
    Peak_potential = 40.
    Normalization = 20.
    Resistance = 10.

    def __init__(self, *, neuron_index: int, column_index: int, type_: int, initial_condition: float, time_step: float,
                 current: float) -> None:
        self.n_idx = neuron_index
        self.col_idx = column_index
        self.fire_times = list()  # all the neuron firing times are stored in a list for future analysis
        self.type = type_  # excitatory (1) or inhibitory (-1)
        self.on = True  # if the neuron is ready to fire, self.on = True. After firing self.on = False for ref_period
        self.off_till = -1  # timer for the neuron off state
        self.v = initial_condition  # initial condition
        # all incoming inputs are stored in a dictionary of format "timestamp: sum of inputs(timestamp)"
        self.incoming_inputs = dict()
        self.tau = int(QIFNeuron.Ref_period / time_step)  # refractory period in integer time steps
        self.i = current  # external current source

    # adjusts the external current source
    def adjust_current(self, current: float) -> None:
        self.i = current

    # receives inputs from other neurons
    def receive_inputs(self, input_time: int, magnitude: float):
        if input_time < self.off_till:
            # if the neuron will still be off when the input comes, ignore it
            pass
        elif input_time not in self.incoming_inputs:
            # create a "timestamp: input(timestamp)" pair
            self.incoming_inputs[input_time] = magnitude
        else:
            # add input to existing "timestamp"
            self.incoming_inputs[input_time] += magnitude

    # integrates the neuron state variable to the next time step, executes neuron firing
    def propagate_step(self, *, current_time: int, dt: float) -> (float, bool):
        firing = False
        # first check if the neuron is turned on
        if self.on:
            # add inputs and integrate dynamics
            if current_time in self.incoming_inputs:
                self.v += self.incoming_inputs[current_time]
                # wipe inputs for current time so the dictionary doesn't become cluttered
                del self.incoming_inputs[current_time]

            # if peak potential exceeded, record spike
            if self.v > QIFNeuron.Peak_potential:
                self.fire_times.append(current_time)
                firing = True

                # turn neuron off, record time until the neuron is turned back on
                self.on = False
                self.off_till = current_time + self.tau
                # off timer randomly varies by a few time steps to avoid a forced period tau in the system
                self.off_till += default_rng().integers(-2, 3, 1)[0]

            else:
                # integrate the dynamics
                sol = solve_ivp(self.eq_of_state, (0, dt), [self.v])
                self.v = sol.y[0][-1]
                # self.v = integrate(eq_of_state=self.eq_of_state, v=self.v, dt=dt)

        else:
            # reset membrane potential
            if self.v != QIFNeuron.Resting_potential:
                self.v = QIFNeuron.Resting_potential
            # if refractory period is over, turn the neuron back on
            if current_time >= self.off_till:
                self.on = True

        # wipe inputs for current time
        if current_time in self.incoming_inputs:
            del self.incoming_inputs[current_time]

        return self.v, firing

    # returns the derivative of v, according to the quadratic integrate-and-fire neuron model
    def eq_of_state(self, t: float, v: float) -> float:
        # dv/dt = [(v-v_r)^2/N + N*I*R_m] / tau_m
        # ERROR HERE, SHOULD'VE BEEN dv/dt = [(v-v_r)^2/N + I*R_m] / tau_m
        return ((v - QIFNeuron.Resting_potential)**2 / QIFNeuron.Normalization
                + QIFNeuron.Normalization * self.i * QIFNeuron.Resistance) / QIFNeuron.Decay_time


class PSLIFNeuron:
    # Neuron physiological parameters
    Capacitance = 2.
    Ref_period = 10.
    E_L = -76.
    g_L = 0.1
    Peak_potential = 40.
    E_Na = 65.
    g_Na = 0.55
    k = 16.
    V_half = 1.5
    Resting_potential = -65.

    def __init__(self, *, neuron_index: int, column_index: int, type_: int, initial_condition: float, time_step: float,
                 current: float) -> None:
        self.n_idx = neuron_index
        self.col_idx = column_index
        self.fire_times = list()  # all the neuron firing times are stored in a list for future analysis
        self.type = type_  # excitatory (1) or inhibitory (-1)
        self.on = True  # if the neuron is ready to fire, self.on = True. After firing self.on = False for ref_period
        self.off_till = -1  # timer for the neuron off state
        self.v = initial_condition  # initial condition
        # all incoming inputs are stored in a dictionary of format "timestamp: sum of inputs(timestamp)"
        self.incoming_inputs = dict()
        self.tau = int(PSLIFNeuron.Ref_period / time_step)  # refractory period in integer time steps
        self.i = current  # external current source

    # adjusts the external current source
    def adjust_current(self, current: float) -> None:
        self.i = current

    # receives inputs from other neurons
    def receive_inputs(self, input_time: int, magnitude: float):
        if input_time < self.off_till:
            # if the neuron will still be off when the input comes, ignore it
            pass
        elif input_time not in self.incoming_inputs:
            # create a "timestamp: input(timestamp)" pair
            self.incoming_inputs[input_time] = magnitude
        else:
            # add input to existing "timestamp"
            self.incoming_inputs[input_time] += magnitude

    # integrates the neuron state variable to the next time step, executes neuron firing
    def propagate_step(self, *, current_time: int, dt: float) -> (float, bool):
        # first check if the neuron is turned on
        if self.on:
            # add inputs and integrate dynamics
            if current_time in self.incoming_inputs:
                self.v += self.incoming_inputs[current_time]
                # wipe inputs for current time so the dictionary doesn't become cluttered
                del self.incoming_inputs[current_time]

            # if divergence threshold exceeded, set potential to amplitude and record spike
            if self.v > PSLIFNeuron.Peak_potential:
                self.fire_times.append(current_time)

                # turn neuron off, record time until the neuron is turned back on
                self.on = False
                self.off_till = current_time + self.tau
                # off timer randomly varies by a few time steps to avoid a forced period tau in the system
                self.off_till += default_rng().integers(-2, 3, 1)[0]

                return self.v, True

            # integrate the dynamics
            sol = solve_ivp(self.eq_of_state, (0, dt), [self.v])
            self.v = sol.y[0][-1]

        else:
            # reset membrane potential
            if self.v != PSLIFNeuron.Resting_potential:
                self.v = PSLIFNeuron.Resting_potential
            # if refractory period is over, turn the neuron back on
            if current_time >= self.off_till:
                self.on = True

        # wipe inputs for current time
        if current_time in self.incoming_inputs:
            del self.incoming_inputs[current_time]

        return self.v, False

    # returns the derivative of v, according to the persistent sodium leaky integrate-and-fire neuron model
    def eq_of_state(self, t: float, v: float) -> float:
        return (-PSLIFNeuron.g_Na * (v - PSLIFNeuron.E_Na) / (1 + np.exp((PSLIFNeuron.V_half - v) / PSLIFNeuron.k))
                - PSLIFNeuron.g_L * (v - PSLIFNeuron.E_L) + self.i) / PSLIFNeuron.Capacitance
