import numpy as np


class InternalConnection:
    """
    Class that contains the targets of a neuron along with their associated connection delays and coupling strengths
    """

    # initializes the instance with connections inside a column
    def __init__(self, *, neuron_index: int, targets: np.ndarray, delays: np.ndarray) -> None:
        self.idx = neuron_index
        self.targets = targets
        self.delays = delays
        # self.gs = coupling_strengths

    # adds targets in neighbouring columns to the instance
    def initialise_out_targets(self, *, targets: np.ndarray, delays: np.ndarray) -> None:
        self.targets = np.concatenate((self.targets, targets))
        self.delays = np.concatenate((self.delays, delays))
        # self.gs = np.concatenate((self.gs, coupling_strengths))
