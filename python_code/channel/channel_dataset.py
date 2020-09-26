import collections
import concurrent.futures
import numpy as np
import torch
from numpy.random import mtrand
from torch.utils.data import Dataset
from typing import Tuple, List
from python_code.channel.channel import ISIAWGNChannel
from python_code.channel.modulator import BPSKModulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORKERS = 16


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words.
    """

    def __init__(self, transmission_length: int,
                 batch_size: int,
                 snr_range: np.ndarray,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 use_ecc: bool):

        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.modulation = BPSKModulator
        self.channel = ISIAWGNChannel
        self.batch_size = batch_size
        self.snr_range = snr_range
        self.memory_length = 4
        if use_ecc:
            self.encoding = lambda b: (np.dot(b, code_gm) % 2)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: int, database: list):
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
            # modulation
            s = self.modulation.modulate(c)
            # transmit through noisy channel
            y = self.channel.transmit(s=s, SNR=snr, random=self.random)

            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            c_full = np.concatenate((c_full, c), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)

        database.append((b_full[:self.batch_size], c_full[:self.batch_size], y_full[:self.batch_size]))

    def __getitem__(self, snr_ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.snr_range, collections.Iterable):
            self.snr_range = [self.snr_range]
        if not isinstance(snr_ind, slice):
            snr_ind = [snr_ind]
        database = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            {executor.submit(self.get_snr_data, snr, database) for snr in self.snr_range[snr_ind]}
        b, c, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.batch_size * len(self.snr_range)
