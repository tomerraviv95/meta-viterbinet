from numpy.random import mtrand
from scipy import signal
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_SIGMA = 1


class ISIAWGNChannel:
    @staticmethod
    def transmit(s: np.ndarray, random: mtrand.RandomState, snr: float, h: np.ndarray, memory_length: int):
        """
        The AWGN Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param random: random words generator
        :param h: channel function
        :param memory_length: length of channel memory
        :return: received word
        """
        snr_value = 10 ** (snr / 10)

        blockwise_s = np.concatenate([s[:, i:-memory_length + i] for i in range(memory_length)], axis=0)

        conv = np.dot(h[:, ::-1], blockwise_s)

        [row, col] = conv.shape

        w = (snr_value ** (-0.5)) * random.normal(0, W_SIGMA, (row, col))

        y = conv # + w

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
