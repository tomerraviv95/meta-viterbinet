import collections
import concurrent.futures
import itertools

import numpy as np
import torch
from numpy.random import mtrand
from torch.utils.data import Dataset
from typing import Tuple, List
from python_code.channel.channel import ISIAWGNChannel, PoissonChannel
from python_code.channel.modulator import BPSKModulator, OnOffModulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORKERS = 16


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words.
    """

    def __init__(self, channel_type: str,
                 transmission_length: int,
                 batch_size: int,
                 snr_range: np.ndarray,
                 gamma_range: np.ndarray,
                 memory_length: int,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 use_ecc: bool):

        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.batch_size = batch_size
        self.snr_range = snr_range
        self.gamma_range = gamma_range
        self.memory_length = memory_length
        if use_ecc:
            self.encoding = lambda b: (np.dot(b, code_gm) % 2)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: int, gamma: int, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.transmission_length))
        c_full = np.empty((0, self.transmission_length))
        y_full = np.empty((0, self.transmission_length + self.memory_length))
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.batch_size:
            # random word generation
            # generate word
            b = self.word_rand_gen.randint(0, 2, size=(self.batch_size, self.transmission_length))
            # encoding - errors correction code
            c = self.encoding(b)

            if self.channel_type == 'ISI_AWGN':
                # modulation
                s = BPSKModulator.modulate(c)
                # transmit through noisy channel
                y = ISIAWGNChannel.transmit(s=s, SNR=snr, random=self.random, gamma=gamma,
                                            memory_length=self.memory_length)
            elif self.channel_type == 'Poisson':
                # modulation
                s = OnOffModulator.modulate(c)
                # transmit through noisy channel
                y = PoissonChannel.transmit(s=s, SNR=snr, random=self.random, gamma=gamma,
                                            memory_length=self.memory_length)
            else:
                raise Exception('No such channel defined!!!')

            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            c_full = np.concatenate((c_full, c), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)

        database.append((b_full[:self.batch_size], c_full[:self.batch_size], y_full[:self.batch_size]))

    def __getitem__(self, snr_ind: int, gamma_ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.snr_range, collections.Iterable):
            self.snr_range = [self.snr_range]
        if not isinstance(self.gamma_range, collections.Iterable):
            self.gamma_range = [self.gamma_range]
        if not isinstance(snr_ind, slice):
            snr_ind = [snr_ind]
        if not isinstance(gamma_ind, slice):
            gamma_ind = [gamma_ind]
        database = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            {executor.submit(self.get_snr_data, snr, gamma, database) for snr, gamma in
             itertools.product(self.snr_range[snr_ind], self.gamma_range[gamma_ind])}
        b, c, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.batch_size * len(self.snr_range) * len(self.gamma_range)
