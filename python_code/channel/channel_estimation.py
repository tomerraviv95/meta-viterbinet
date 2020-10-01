import numpy as np


def estimate_channel(memory_length: int, gamma: float, noisy_est_var: float = 0):
    h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
    if noisy_est_var > 0:
        h[:, 1:] += np.random.normal(0, noisy_est_var ** 0.5, [1, memory_length - 1])
    return h
