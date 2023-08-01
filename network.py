from column import Column
from connections import InternalConnection
import multiprocessing as mp
from neuron import LIFNeuron, QIFNeuron, PSLIFNeuron
from numba import njit
import numpy as np
from numpy.random import default_rng, poisson
from support import construct_neighbours_dictionary, smart_round
import matplotlib.pyplot as plt


class Network:

    # Network physiological parameters
    Column_size = 10000  # num of neurons per column
    Synapses_per_neuron = 2300  # avg num of connections each neuron forms
    Synapses_HWHM = 230  # HWHM of the synapse number distribution
    Inter_col_synapses = 100  # avg num of connections each inter-columnar neuron forms with each neighbouring column
    ICS_HWHM = 10  # HWHM of the inter column synapse number distribution
    Min_delay = 2.5  # typical synaptic transmission delay in ms
    Min_delay_spread = 0.4  # spread of synaptic transmission delay around the typical value in log-normal dist. in ms
    Neighbours = {0: []}  # indices of neighbouring columns in a columnar network
    Layer23neurons = 0.28
    Connection_speed = 1.  # typical transmission speed in mm/ms
    Speed_spread = 0.3  # spread of transmission speed around the typical value in log-normal dist. in mm/ms

    def __init__(self, *, size: int, e_i_balance: float, time_step: float, g_e: float, g_i: float, current: float,
                 model: str, filepath: str, ic_mu: float = -55., ic_sig: float = 10.,
                 ic_array: np.ndarray or None = None, col_size: int = Column_size) -> None:
        """

        :param size:            integer. Number of columns in the network
        :param e_i_balance:     float. Percentage of excitatory neurons in a column
        :param time_step:       float. Simulation time step in real time in units 1 ms.
        :param g_e:             float. Coupling strength of excitatory neurons.
        :param g_i:             float. Coupling strength of inhibitory neurons.
        :param current:         float. Current due to external activity.
        :param model:           string. Model to be used for neuron dynamics (LIF, QIF or PSLIF).
        :param filepath:        string. Path to the directory in which to store the model.
        :param ic_mu:           float. Neuron initial condition distribution average. Default value -55., halfway
                                between rest (-65.) and threshold (-45.).
        :param ic_sig:          float. Neuron initial condition distribution width. Default value 10.
        :param ic_array:        np.ndarray or None. Predetermined neuron initial condition distribution. Default None.
        :param col_size:        int. Number of neurons in column. Default value Column_size (10000).

        """
        # proportion of excitatory neurons in layers II and III
        self.e_i_balance = e_i_balance
        self.layer23neurons = Network.Layer23neurons / e_i_balance

        # construct dictionary of column neighbour indices
        if size > 1:
            Network.Neighbours = construct_neighbours_dictionary(size)

        # fundamental attributes
        self.num_of_columns = size
        self.num_of_neurons = size * col_size  # can be adapted for heterogeneous network later

        # metrics
        self.avg_potential = {-1: [], 1: [], 0: []}
        self.firing_rate = {-1: [], 1: [], 0: []}

        # timer
        self.current_time = 0  # counts the integer time step the network is on
        self.dt = time_step  # the size of each time step in real time [ms]

        # neurons
        self.neuron_model = model
        self.neurons = self._initialise_neurons(current, ic_mu, ic_sig, ic_array)  # shape (size, Column_size)

        # connections
        self.connections = self._initialise_connections(time_step)  # shape (size, Column_size, variable))
        self.couplings = {-1: -g_i, 1: g_e}  # connections couplings

        # columns for semi-local metrics
        self.columns = np.empty(size, dtype=object)
        for i in range(size):
            self.columns[i] = Column(index=i, size=Network.Column_size, e_i_balance=e_i_balance)

        # filepath for storing data
        self.path = filepath

    # Propagate the network to the next time step.
    #
    # First get all the external inputs from the external drive, then loop over all columns and all neurons in a column.
    # Integrate the neuron's membrane potential according to internal dynamics. If the neuron fires, send outputs.
    # Else, send external inputs from the background and the external drive to the neuron. Update metrics throughout.
    def propagate_step(self) -> None:
        v = {-1: 0, 1: 0}  # running sum for network voltage
        r = {-1: 0, 1: 0}  # running sum for network firing rate

        for i, c in enumerate(self.neurons):
            v_c, r_c = self.propagate_step_on_column(c, i)

            # update column metrics and global running sums
            self.columns[i].update_metrics(avg_potential=v_c, firing_rate=r_c)
            for type_ in (-1, 1):
                v[type_] += v_c[type_]
                r[type_] += r_c[type_]

        # pool = mp.Pool(self.max_cores)
        #
        # # Execute the propagation tasks in parallel
        # info = pool.starmap(self.propagate_step_on_column, iterable=[
        #     (c, i, stimulus_targets_left, stimulus_targets_right, g) for i, c in enumerate(self.neurons)])
        # pool.close()
        # pool.join()
        #
        # for i, (v_c, r_c) in enumerate(info):
        #     # update column metrics and global running sums
        #     self.columns[i].update_metrics(avg_potential=v_c, firing_rate=r_c)
        #     v += v_c[-1] + v_c[1]
        #     r += r_c[-1] + r_c[1]

        # update current time and global metrics
        self.current_time += 1
        for type_ in (-1, 1):
            self.avg_potential[type_].append(v[type_] / (
                    ((1 - type_) / 2 + type_ * self.e_i_balance) * self.num_of_neurons))
            self.firing_rate[type_].append(r[type_])
        self.avg_potential[0].append((v[-1] + v[1]) / self.num_of_neurons)
        self.firing_rate[0].append(r[-1] + r[1])

    def propagate_step_on_column(self, column: np.ndarray, i: int) -> (dict, dict):
        v_c = {-1: 0., 1: 0.}  # running sums for neuron-type-specific voltage
        r_c = {-1: 0, 1: 0}  # and firing rate for the given column
        for j, n in enumerate(column):
            # update the state of neuron n and add it to the running sum v_c
            dv, firing = n.propagate_step(current_time=self.current_time, dt=self.dt)
            v_c[n.type] += dv

            # send outputs of this neuron to connected neurons and update firing rate r_c
            if firing:
                r_c[n.type] += 1

                for k, [col_idx, neuron_idx] in enumerate(self.connections[i, j].targets):
                    self.neurons[col_idx, neuron_idx].receive_inputs(
                        input_time=self.current_time + self.connections[i, j].delays[k],
                        magnitude=self.couplings[n.type])

        return v_c, r_c

    def adjust_current(self, *, col_idx: int or None, current: float) -> None:
        import datetime
        if col_idx is None:  # adjust current in all neurons
            with open(self.path + '/README.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime(
                    "%Y-%m-%d %X : adjusted current to all columns to {}").format(current) + '\n')
            for column in self.neurons:  # loop over columns
                for neuron in column:  # loop over neurons in a column
                    neuron.adjust_current(current)
        else:  # adjust current only in neurons of specified column
            with open(self.path + '/README.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime(
                    "%Y-%m-%d %X : adjusted current to column {} to {}").format(col_idx, current) + '\n')
            column = self.neurons[col_idx]
            for neuron in column:  # loop over neurons in column
                neuron.adjust_current(current)

    # Save the Network object in parts
    def save_network(self, *, description: str or None = None, overwrite: bool = False,
                     return_timer: bool = False, write_only: bool = False) -> None or float:
        import datetime
        import os
        import pickle
        import time
        t0 = time.time()
        if write_only:
            assert os.path.exists(self.path)
            if description is not None:
                with open(self.path + '/README.txt', 'a') as f:
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d %X : ") + description + '\n')
        else:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
                pickle.dump(self, open(self.path + '/net.pkl', 'wb'))
            elif overwrite:
                pickle.dump(self, open(self.path + '/net.pkl', 'wb'))
            if description is not None:
                with open(self.path + '/README.txt', 'a') as f:
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d %X : ") + description + '\n')
            pickle.dump(self.neurons, open(self.path + '/neurons.pkl', 'wb'))
            pickle.dump(self.avg_potential, open(self.path + '/avg_potential.pkl', 'wb'))
            pickle.dump(self.columns, open(self.path + '/columns.pkl', 'wb'))
            pickle.dump(self.current_time, open(self.path + '/current_time.pkl', 'wb'))
            pickle.dump(self.firing_rate, open(self.path + '/firing_rate.pkl', 'wb'))
        dt = time.time() - t0
        print(f'network saved in:  {dt: .2f}')
        if return_timer:
            return dt

    # construct a (num_of_columns, Column_size) shaped array of SpikingNeuron objects
    def _initialise_neurons(self, current: float, mu: float, sig: float, arr: np.ndarray or None) -> np.ndarray:
        neurons = np.empty((self.num_of_columns, Network.Column_size), dtype=object)
        rng = default_rng()

        for column in range(self.num_of_columns):  # loop over columns
            type_ = 1  # start with excitatory neurons
            e_i_split = int(self.e_i_balance * Network.Column_size)

            if arr is None:
                arr = rng.normal(mu, sig, Network.Column_size)

            for j in range(Network.Column_size):  # loop over neurons in a column
                # switch to inhibitory neurons, once all excitatory ones are constructed
                if type_ == 1 and j >= e_i_split:
                    type_ = -1

                # generate initial state and construct neuron
                ic = arr[j]
                neurons[column, j] = eval(
                    self.neuron_model + 'Neuron(neuron_index=j, column_index=column, type_=type_, initial_condition=ic,'
                                        'time_step=self.dt, current=current)')

        return neurons

    # construct a (num_of_columns, Column_size) shaped array of InternalConnection objects
    def _initialise_connections(self, dt: float) -> np.ndarray:
        connections = np.empty((self.num_of_columns, Network.Column_size), dtype=object)
        rng = default_rng()  # initialise random number generator
        axonal_lengths = np.load('data/extras/distances.npy')
        axonal_lengths_external = np.load('data/extras/neighbour_distances.npy')

        for i, c in enumerate(self.neurons):  # loop over columns

            for j, n in enumerate(c):  # loop over neurons in a column

                if n.type == 1:  # excitatory neurons; sometimes have connections to neighbouring columns

                    if self.num_of_columns > 1 and rng.random() <= self.layer23neurons:
                        # neuron n is from layer II or III
                        inter_column_connections = True
                    else:
                        inter_column_connections = False

                    # number of connections neuron n forms (gaussian with mean Synapses_per_neuron)
                    k = int(rng.normal(Network.Synapses_per_neuron, Network.Synapses_HWHM, 1)[0])

                    # all possible targets inside column c
                    temp = np.concatenate((np.arange(j), np.arange(j + 1, Network.Column_size)), dtype=int)
                    # targets are specified by a doublet [column_index, neuron_index]
                    targets = np.empty((k, 2), dtype=int)
                    targets[:, 0] = i
                    # draw k samples from all possible targets inside column
                    targets[:, 1] = rng.choice(temp, k)

                    # draw k random delays (distribution of distances in column) for targets inside column
                    speeds = rng.lognormal(np.log(Network.Connection_speed), Network.Speed_spread, k)
                    synaptic_delays = rng.lognormal(np.log(Network.Min_delay), Network.Min_delay_spread, k) / dt
                    axonal_delays = rng.choice(axonal_lengths, k) / speeds / dt
                    delays = (np.round(synaptic_delays + axonal_delays, 0)).astype(int)

                    # plt.hist(synaptic_delays, bins=31, label='synaptic', alpha=0.5)
                    # plt.hist(axonal_delays, bins=31, label='axonal', alpha=0.5)
                    # plt.hist(delays, bins=31, label='together', alpha=0.5)
                    # plt.legend()
                    # plt.grid()
                    # plt.show()

                    # initialise connection
                    connections[i, j] = InternalConnection(neuron_index=j, targets=targets, delays=delays)

                    # if the connection belongs to a neuron from layers II or III,
                    # add connections with neighbouring columns
                    if inter_column_connections:
                        # find neighbouring columns of column c (with index i)
                        neighbours = Network.Neighbours[i]
                        # number of connections to neighbouring columns
                        # (gaussian with mean number_of_neighbouring_columns * Ics_pn_pc)
                        k_ext = int(rng.normal(len(neighbours) * Network.Inter_col_synapses, Network.ICS_HWHM, 1)[0])

                        # all possible targets in neighbouring columns
                        temp = np.zeros((len(neighbours) * Network.Column_size, 2), dtype=int)
                        for m, col in enumerate(neighbours):
                            temp[m * Network.Column_size:(m + 1) * Network.Column_size, 0] = col
                            temp[m * Network.Column_size:(m + 1) * Network.Column_size, 1] = \
                                np.arange(Network.Column_size)
                        # draw k_ext samples from all possible targets to neighbouring columns
                        targets_ext = rng.choice(temp, k_ext)

                        # draw k_ext random delays (distribution of distances) for targets in neighbouring columns
                        speeds = rng.lognormal(np.log10(Network.Connection_speed), Network.Speed_spread, k_ext)
                        synaptic_delays = rng.lognormal(np.log(Network.Min_delay), Network.Min_delay_spread, k_ext) / dt
                        axonal_delays = rng.choice(axonal_lengths_external, k_ext) / speeds / dt
                        delays_ext = (np.round(synaptic_delays + axonal_delays, 0)).astype(int)

                        # update connection with targets in neighbouring columns
                        connections[i, j].initialise_out_targets(targets=targets_ext, delays=delays_ext)

                elif n.type == -1:  # inhibitory neurons, only have intra-column connections
                    # number of connections neuron n forms (gaussian with mean Synapses_per_neuron)
                    k = int(rng.normal(Network.Synapses_per_neuron, Network.Synapses_HWHM, 1)[0])

                    # all possible targets inside column c
                    temp = np.concatenate((np.arange(j), np.arange(j + 1, Network.Column_size)), dtype=int)
                    # targets are specified by a doublet [column_index, neuron_index]
                    targets = np.empty((k, 2), dtype=int)
                    targets[:, 0] = i
                    # draw k samples from all possible targets inside column
                    targets[:, 1] = rng.choice(temp, k)

                    # draw k random delays (distribution of distances in column) for targets inside column
                    speeds = rng.lognormal(np.log10(Network.Connection_speed), Network.Speed_spread, k)
                    synaptic_delays = rng.lognormal(np.log(Network.Min_delay), Network.Min_delay_spread, k) / dt
                    axonal_delays = rng.choice(axonal_lengths, k) / speeds / dt
                    delays = (np.round(synaptic_delays + axonal_delays, 0)).astype(int)

                    # initialise connection
                    connections[i, j] = InternalConnection(neuron_index=i, targets=targets, delays=delays)

        return connections


# Load a Network object in parts
def load_network(*, filepath: str, show_readme: bool = True, return_timer: bool = False) -> Network or (Network, float):
    from os.path import exists
    import pickle
    import time
    t0 = time.time()
    if show_readme:
        if exists(filepath + '/README.txt'):
            with open(filepath + '/README.txt', 'r') as f:
                for line in f.readlines():
                    print(*line.rsplit('\n'))
    net = pickle.load(open(filepath + '/net.pkl', 'rb'))
    net.neurons = pickle.load(open(filepath + '/neurons.pkl', 'rb'))
    net.avg_potential = pickle.load(open(filepath + '/avg_potential.pkl', 'rb'))
    net.columns = pickle.load(open(filepath + '/columns.pkl', 'rb'))
    net.current_time = pickle.load(open(filepath + '/current_time.pkl', 'rb'))
    net.firing_rate = pickle.load(open(filepath + '/firing_rate.pkl', 'rb'))
    dt = time.time() - t0
    if show_readme:
        print(f'network loaded in:  {dt: .2f}')
    if return_timer:
        return net, dt
    else:
        return net


def reconstruct_population_specific_global_data(net: Network) -> (dict, dict):
    avg_potential = {-1: 0, 1: 0, 0: np.array(net.avg_potential)}
    firing_rate = {-1: 0, 1: 0, 0: np.array(net.firing_rate)}
    for c in net.columns:
        avg_potential[-1] += np.array(c.avg_potential[-1]) / net.num_of_columns
        avg_potential[1] += np.array(c.avg_potential[1]) / net.num_of_columns
        firing_rate[-1] += np.array(c.firing_rate[-1]) / net.num_of_columns
        firing_rate[1] += np.array(c.firing_rate[1]) / net.num_of_columns
    for i in (-1, 1, 0):
        avg_potential[i] = avg_potential[i].tolist()
        firing_rate[i] = firing_rate[i].tolist()
    return avg_potential, firing_rate


