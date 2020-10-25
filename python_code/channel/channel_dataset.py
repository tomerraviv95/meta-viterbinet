import concurrent.futures
import random

import numpy as np
import torch
from numpy.random import mtrand
from torch.utils.data import Dataset
from typing import Tuple, List
from python_code.channel.channel import ISIAWGNChannel, PoissonChannel
from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator, OnOffModulator
from python_code.ecc.rs_main import encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, channel_type: str,
                 block_length: int,
                 transmission_length: int,
                 channel_blocks: int,
                 words: int,
                 memory_length: int,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 noisy_est_var: float,
                 use_ecc: bool,
                 n_symbols: int,
                 fading_in_channel: bool,
                 fading_in_decoder: bool,
                 phase: str):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.channel_blocks = channel_blocks
        self.words = words
        self.memory_length = memory_length
        self.noisy_est_var = noisy_est_var
        self.fading_in_channel = fading_in_channel
        self.fading_in_decoder = fading_in_decoder
        self.n_symbols = n_symbols
        self.phase = phase
        if use_ecc and phase == 'val':
            self.encoding = lambda b: encode(b, self.n_symbols)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: float, gamma: float, database: list):
        if database is None:
            database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        if self.phase == 'val':
            index = 0
        else:
            index = random.randint(0, 1e6)
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            # generate word
            b = self.word_rand_gen.randint(0, 2, size=(1, self.block_length))
            # encoding - errors correction code
            c = self.encoding(b).reshape(1, -1)
            # add zero bits
            padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
            # transmit - validation
            if self.phase == 'val':
                # channel_estimate
                h = estimate_channel(self.memory_length, gamma, fading=self.fading_in_channel, index=index)
                y = self.transmit(padded_c, h, snr)
            # transmit - training
            elif self.phase == 'train' or self.phase == 'meta_train':
                y = np.zeros(b.shape)
                assert self.channel_blocks == 1  # only one channel for one local update
                block_length = self.transmission_length // self.channel_blocks
                for channel_block in range(self.channel_blocks):
                    block_start = channel_block * block_length
                    block_end = (channel_block + 1) * block_length
                    h = estimate_channel(self.memory_length, gamma, noisy_est_var=self.noisy_est_var,
                                         fading=self.fading_in_decoder, index=index)
                    y[:, block_start: block_end] = self.transmit(
                        padded_c[:, block_start: block_end + self.memory_length], h,
                        snr)
            else:
                raise NotImplementedError("No such phase implemented!!!")

            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            index += 1

        database.append((b_full, y_full))

    def transmit(self, c: np.ndarray, h: np.ndarray, snr: float):
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
        return y

    def __getitem__(self, snr_list: List[float], gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, gamma, database) for snr in snr_list]
        b, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.transmission_length * self.channel_blocks
