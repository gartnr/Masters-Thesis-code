import numpy as np
import scipy.integrate
from scipy.optimize import fsolve
from scipy.signal import convolve
from scipy.signal.windows import hann


def smart_round(a, d):  # currently not in use
    return np.round(a / d, 0) * d


def firing_rate_ma(r: np.ndarray, start: int, length: int, period: int = 10) -> np.ndarray:
    assert start - period // 2 >= 0
    assert start + period // 2 + 1 <= len(r)
    ma = np.asarray([np.sum(r[i:i+period]) / period for i in range(start - period // 2, start - period // 2 + length)])
    return ma


def smoothen(signal: np.ndarray, period: int = 100, with_averaging: bool = True) -> np.ndarray:
    win = hann(period)
    if with_averaging:
        a = np.average(signal)
        smooth_signal = convolve(signal - a, win, 'same') / sum(win) + a
    else:
        smooth_signal = convolve(signal, win, 'same') / sum(win)
    return smooth_signal


def qif_period(current: np.ndarray or float, N: float = 20., R_m: float = 10 , E_L: float = -65., V_AP: float = 40.,
               tau: float = 20.) -> np.ndarray or float:
    return tau * np.sqrt(N / (R_m * current)) * np.arctan((V_AP - E_L) / np.sqrt(R_m * current * N))


def lif_period(current: np.ndarray or float, R_m: float = 10 , E_L: float = -65., theta: float = -45., tau: float = 20.,
               ) -> np.ndarray or float:
    return tau * np.log((R_m * current) / (-theta + E_L + R_m * current))


def pslif_period(current: np.ndarray, g_Na: float = 0.55, g_L: float = 0.1, E_L: float = -76., k: float = 16,
                 E_Na: float = 65., V_r: float = -65., V_ap: float = 40., C_m: float = 2., V_half: float = 1.5
                 ) -> np.ndarray or float:

    sol = np.zeros_like(current)

    def m(V):
        return 1 / (1 + np.exp((V_half - V) / k))

    def f(V, I):
        return 1 / ((-g_L * (V - E_L) - g_Na * m(V) * (V - E_Na) + I) / C_m)

    for i, c in enumerate(current):
        sol[i] = scipy.integrate.quad(f, V_r, V_ap, args=c)[0]

    return sol


def single_neuron_firing_rate(fire_times: list, start: int, end: int) -> float:
    assert end > start
    r = 0.
    for t in fire_times:
        if start <= t <= end:
            r += 1
    return r / (end - start)


def single_neuron_ISI(fire_times: list, start: int, end: int) -> np.ndarray:
    assert end > start
    ft = np.array(fire_times)
    ISIs = ft[1:] - ft[:-1]
    return ISIs


def construct_neighbours_dictionary(n: int) -> dict:
    """
    The studied network is a hexagonal columnar structure made up of n columns. The hexagons (columns) are arranged in
    concentric hexagonal rings, where ring i has 6 * i hexagons, 6 of them corner hexagons and 6 * (i - 1) edge
    hexagons. Ring 0 is a special case. It contains a single hexagon with index 0.

        >construct_neighbours_dictionary(network_size) constructs a dictionary of lists
        >each key is the index of a network column
        >the list corresponding to some key (column) contains the indices of all neighbouring columns in ascending order

    :param n:       number of columns in the network
    :return:        dictionary of lists of the form `column_index`: [neighbour_1_index, ..., neighbour_k<=6_index]
    """

    if n == 1:

        return {0: [], }

    else:

        # for the time being, neighbours_dict is a dictionary of dictionaries, so that indices don`t repeat
        neighbours_dict = {0: {}}
        ring_counter = 1
        hexagon_counter = 6
        internal_counter = 0

        for i in range(1, n):

            if i <= 6:  # special case: "ring 0" that contains only column 0
                neighbours_dict[0][i] = 0

            # neighbours of hexagon i are always i - 1 and i + 1 (if i + 1 exists)
            neighbours_dict[i] = {i - 1: 0}
            neighbours_dict[i - 1][i] = 0

            if i > hexagon_counter:  # moving to the next ring
                internal_counter = 0
                ring_counter += 1
                hexagon_counter += 6 * ring_counter

            if ring_counter > 1:  # neighbours of column i from previous ring (starts at ring = 2)
                temp = i - 6 * (ring_counter - 1) - internal_counter  # index of closes neighbour from previous ring
                if i % ring_counter == 0:
                    # corner hexagon: has 1 neighbour from previous ring
                    internal_counter += 1
                    neighbours_dict[i][temp - 1] = 0
                else:
                    # edge hexagon: has 2 neighbours from previous ring
                    neighbours_dict[i][temp] = 0
                    if i != hexagon_counter - 6 * ring_counter + 1:
                        neighbours_dict[i][temp - 1] = 0
                        # first column in new ring has index (hexagon_counter - 6 * ring_counter) and one of its
                        # neighbours is (i - 1), not (temp - 1)
            if i == hexagon_counter:  # connects the first and last column in a ring
                neighbours_dict[hexagon_counter - 6 * ring_counter + 1][hexagon_counter] = 0
                neighbours_dict[hexagon_counter][hexagon_counter - 6 * ring_counter + 1] = 0

        # go through the constructed network and make sure, that if i is the neighbour of j, j is also the neighbour
        # of i basically mirror the previous ring neighbours as next ring neighbours
        for i in range(n):
            for j in neighbours_dict[i].keys():
                neighbours_dict[j][i] = 0

        # change the inside dictionaries into ordered lists
        for key, sub_dict in neighbours_dict.items():
            a = list(sub_dict.keys())
            a.sort()
            neighbours_dict[key] = a
        return neighbours_dict


def draw_left_right_columns(centers_x: np.ndarray, r: float) -> np.ndarray:
    identities = np.zeros_like(centers_x)
    rng = np.random.default_rng()
    for i, x in enumerate(centers_x):
        if rng.random(1)[0] < r + (1 - 2 * r) * (abs(round(x, 0)) % 2):
            identities[i] = 1
    return identities


def calculate_ul_ur(a: float) -> (float, float):
    """
    :param a:       key parameter of the equation of state: u` = -u + exp(u + a)
    :return:        solutions (0 < u_L < 1, u_R > 1) to u` = 0. These solutions exist for exp(a) < 1/e => a < -1.
    """
    assert a < -1

    def eq_of_state(u):
        return np.exp(u + a) - u

    u_L = fsolve(eq_of_state, np.array([0]))[0]
    u_R = fsolve(eq_of_state, np.array([10]))[0]
    return u_L, u_R
