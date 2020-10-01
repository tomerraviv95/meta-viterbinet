import collections
import concurrent.futures
import itertools

import numpy as np
import torch
from numpy.random import mtrand
from torch.utils.data import Dataset
from typing import Tuple, List
from python_code.channel.channel import ISIAWGNChannel, PoissonChannel
from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator, OnOffModulator
from python_code.utils.numpy_utils import shift_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORKERS = 16


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words.
    """

    def __init__(self, channel_type: str,
                 transmission_length: int,
                 channel_blocks: int,
                 memory_length: int,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 use_ecc: bool):

        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.channel_blocks = channel_blocks
        self.memory_length = memory_length
        if use_ecc:
            self.encoding = lambda b: (np.dot(b, code_gm) % 2)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: int, gamma: int, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.transmission_length))
        y_full = np.empty((0, self.transmission_length))
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.channel_blocks:
            # random word generation
            # generate word
            b = self.word_rand_gen.randint(0, 2, size=(1, self.transmission_length))
            # add zero bits
            padded_b = np.concatenate([b, np.zeros([b.shape[0], self.memory_length])], axis=1)
            # encoding - errors correction code
            c = self.encoding(padded_b)
            # channel_estimate
            h = estimate_channel(self.memory_length, gamma)
            if self.channel_type == 'ISI_AWGN':
                # modulation
                s = BPSKModulator.modulate(c)
                # transmit through noisy channel
                y = ISIAWGNChannel.transmit(s=s, random=self.random, h=h, snr=snr, memory_length=self.memory_length)
            elif self.channel_type == 'Poisson':
                # modulation
                s = OnOffModulator.modulate(c)
                # transmit through noisy channel
                y = PoissonChannel.transmit(s=s, random=self.random, h=h, memory_length=self.memory_length)
            else:
                raise Exception('No such channel defined!!!')

            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)

        database.append((b_full, y_full))

    def __getitem__(self, snr: float, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
        database = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.submit(self.get_snr_data, snr, gamma, database)
        b, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.transmission_length * self.channel_blocks
