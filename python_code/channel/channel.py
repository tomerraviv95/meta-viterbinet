from numpy.random import mtrand
from scipy import signal
import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_SIGMA = 1


class ISIAWGNChannel:
    @staticmethod
    def transmit(s: np.ndarray, SNR: int, random: mtrand.RandomState, gamma: float, memory_length: int):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """

        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])

        SNR_value = 10 ** (SNR / 10)

        padded_s = np.concatenate([np.zeros([s.shape[0], memory_length + 1]), s, np.ones([s.shape[0], memory_length])],
                                  axis=1)

        conv_out = signal.convolve2d(padded_s, h, 'same')[:, memory_length:-1]

        [row, col] = conv_out.shape

        w = random.normal(0, W_SIGMA, (row, col))

        h_tilde = h.copy()
        h_tilde[:,0] = 0
        isi = signal.convolve2d(padded_s, h_tilde, 'same')[:, memory_length:-1]
        conv_out_tilde = (conv_out - isi)

        y = math.sqrt(SNR_value) * conv_out + w

        return y


class PoissonChannel:
    @staticmethod
    def transmit(s: np.ndarray, SNR: int, random: mtrand.RandomState, gamma: float, memory_length: int):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """

        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])

        SNR_value = 10 ** (SNR / 10)

        padded_s = np.concatenate([np.zeros([s.shape[0], memory_length + 1]), s, np.zeros([s.shape[0], memory_length])],
                                  axis=1)

        conv_out = signal.convolve2d(padded_s, h, 'same')[:, memory_length:-1]

        y = np.random.poisson(math.sqrt(SNR_value) * conv_out + 1)

        return y
