from dir_definitions import COST2100_DIR
import numpy as np
import scipy.io
import os


def estimate_channel(memory_length: int, gamma: float, channel_coefficients: str, noisy_est_var: float = 0,
                     fading: bool = False, index: int = 0):
    """
    Returns the coefficients vector estimated from channel
    :param memory_length: memory length of channel
    :param gamma: coefficient
    :param channel_coefficients: coefficients type
    :param noisy_est_var: variance for noisy estimation of coefficients 2nd,3rd,...
    :param fading: fading flag - if true, apply fading.
    :param index: time index for the fading functionality
    :return: the channel coefficients [1,memory_length] numpy array
    """
    if channel_coefficients == 'time_decay':
        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
    elif channel_coefficients == 'cost2100':
        total_h = np.empty([100, memory_length])
        for i in range(memory_length):
            total_h[:, i] = scipy.io.loadmat(os.path.join(COST2100_DIR, f'h_{0.02*(i+1)}'))[
                'h_channel_response_mag'].reshape(-1)
        # scale min-max values of h to the range 0-1
        total_h = (total_h - total_h.min()) / (total_h.max() - total_h.min())
        h = np.reshape(total_h[index], [1, memory_length])
    else:
        raise ValueError('No such channel_coefficients value!!!')
    # noise in estimation of h taps
    if noisy_est_var > 0:
        h[:, 1:] += np.random.normal(0, noisy_est_var ** 0.5, [1, memory_length - 1])
    # fading in channel taps
    if fading:
        # fading_taps = np.array([51, 39, 33, 21])
        fading_taps = np.array([40, 32, 25, 15])
        h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
    return h
