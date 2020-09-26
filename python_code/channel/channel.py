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

        memory_size = 4
        gamma = 2
        w_sigma = 1

        h = np.reshape(np.exp(-gamma * np.arange(memory_size)), [1, memory_size])

        SNR_value = 10 ** (SNR / 10)

        conv_out = signal.convolve(s, h)

        [row, col] = conv_out.shape

        w = random.normal(0, w_sigma, (row, col))

        y = math.sqrt(SNR_value) * conv_out + w

        llr = 2 * y / w_sigma ** 2

        return llr
