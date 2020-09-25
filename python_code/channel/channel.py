from numpy.random import mtrand
from scipy import signal
import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ISIAWGNChannel:
    @staticmethod
    def transmit(s: np.ndarray, SNR: int, random: mtrand.RandomState):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """
        [row, col] = s.shape

        memory_size = 4
        gamma = 0.2

        h = np.reshape(np.exp(-gamma * np.arange(memory_size)), [1, memory_size])

        w = random.normal(0.0, 1.0, (row, col))

        y = math.sqrt(SNR) * signal.convolve(s, h) + w

        return y
