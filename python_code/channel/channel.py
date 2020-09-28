from numpy.random import mtrand
from scipy import signal
import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_SIGMA = 1


class ISIAWGNChannel:
    @staticmethod
    def transmit(s: np.ndarray, random: mtrand.RandomState, h: np.ndarray, memory_length: int):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """

        padded_s = np.concatenate([np.zeros([s.shape[0], memory_length + 1]), s, np.ones([s.shape[0], memory_length])],
                                  axis=1)

        conv_out = signal.convolve2d(padded_s, h, 'same')[:, memory_length:-1]

        [row, col] = conv_out.shape

        w = random.normal(0, W_SIGMA, (row, col))

        y = conv_out + w

        return y


class PoissonChannel:
    @staticmethod
    def transmit(s: np.ndarray, random: mtrand.RandomState, h: np.ndarray, memory_length: int):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """

        padded_s = np.concatenate([np.zeros([s.shape[0], memory_length + 1]), s, np.zeros([s.shape[0], memory_length])],
                                  axis=1)

        conv_out = signal.convolve2d(padded_s, h, 'same')[:, memory_length:-1]

        y = np.random.poisson(conv_out + 1)

        return y
