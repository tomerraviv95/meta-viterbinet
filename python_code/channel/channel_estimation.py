import numpy as np


def estimate_channel(memory_length: int, gamma: float, noisy_est_var: float = 0, fading: bool = False, index: int = 0):
    """
    Returns the coefficients vector estimated from channel
    :param memory_length: memory length of channel
    :param gamma: coefficient
    :param noisy_est_var: variance for noisy estimation of coefficients 2nd,3rd,...
    :param fading: fading flag - if true, apply fading.
    :param index: time index for the fading functionality
    :return: the channel coefficients [1,memory_length] numpy array
    """
    h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
    # noise in estimation of h taps
    if noisy_est_var > 0:
        h[:, 1:] += np.random.normal(0, noisy_est_var ** 0.5, [1, memory_length - 1])
    # fading in channel taps
    if fading:
        fading_taps = np.array([51, 39, 33, 21])
        h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
    return h
