import numpy as np


class Column:
    """
    Class that keeps track of column firing rates and average potential for the excitatory and inhibitory population
    """

    def __init__(self, *, index: int, size: int, e_i_balance: float) -> None:
        self.index = index  # column index
        self.e_i_balance = e_i_balance  # proportion of excitatory neurons
        # absolute number of excitatory (1) and inhibitory (-1) neurons
        self.size = {1: int(round(e_i_balance * size, 0)), -1: int(round((1 - e_i_balance) * size, 0))}
        # average column potential
        self.avg_potential = {1: [], -1: []}
        # column firing rate
        self.firing_rate = {1: [], -1: []}

    # updates the average potential and firing rate, is called after every step
    def update_metrics(self, *, avg_potential: dict, firing_rate: dict) -> None:

        for neuron_type in [-1, 1]:
            self.avg_potential[neuron_type].append(avg_potential[neuron_type] / self.size[neuron_type])
            self.firing_rate[neuron_type].append(firing_rate[neuron_type])
