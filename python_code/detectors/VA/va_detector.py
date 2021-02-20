from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
import numpy as np
import torch
import torch.nn as nn
import math

from python_code.utils.trellis_utils import create_transition_table, acs_block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VADetector(nn.Module):
    """
    This module implements the classic VA detector
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_length: int,
                 val_words: int,
                 channel_type: str,
                 noisy_est_var: float,
                 fading: bool,
                 fading_taps_type: int,
                 channel_coefficients: str):

        super(VADetector, self).__init__()
        self.memory_length = memory_length
        self.transmission_length = transmission_length
        self.val_words = val_words
        self.n_states = n_states
        self.channel_type = channel_type
        self.noisy_est_var = noisy_est_var
        self.fading = fading
        self.fading_taps_type = fading_taps_type
        self.channel_coefficients = channel_coefficients
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)

    def compute_state_priors(self, h: np.ndarray) -> torch.Tensor:
        all_states_decimal = np.arange(self.n_states).astype(np.uint8).reshape(-1, 1)
        all_states_binary = np.unpackbits(all_states_decimal, axis=1).astype(int)
        if self.channel_type == 'ISI_AWGN':
            all_states_symbols = BPSKModulator.modulate(all_states_binary[:, -self.memory_length:])
        else:
            raise Exception('No such channel defined!!!')
        state_priors = np.dot(all_states_symbols, h.T)
        return torch.Tensor(state_priors).to(device)

    def compute_likelihood_priors(self, y: torch.Tensor, snr: float, gamma: float, phase: str, count: int = None):
        # estimate channel per word (only changes between the h's if fading is True)
        h = np.concatenate([estimate_channel(self.memory_length, gamma, noisy_est_var=self.noisy_est_var,
                                             fading=self.fading, index=index, fading_taps_type=self.fading_taps_type,
                                             channel_coefficients=self.channel_coefficients[phase]) for index in
                            range(self.val_words)],
                           axis=0)
        if count is not None:
            h = h[count].reshape(1, -1)
        # compute priors
        state_priors = self.compute_state_priors(h)
        if self.channel_type == 'ISI_AWGN':
            priors = y.unsqueeze(dim=2) - state_priors.T.repeat(
                repeats=[y.shape[0] // state_priors.shape[1], 1]).unsqueeze(
                dim=1)
            # to llr representation
            priors = priors ** 2 / 2 - math.log(math.sqrt(2 * math.pi))
        else:
            raise Exception('No such channel defined!!!')
        return priors

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None,
                count: int = None) -> torch.Tensor:
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns tensor of detected word, same shape as y
        """
        # initialize input probabilities
        in_prob = torch.zeros([y.shape[0], self.n_states]).to(device)

        # compute transition likelihood priors
        priors = self.compute_likelihood_priors(y, snr, gamma, phase, count)

        if phase == 'val':
            decoded_word = torch.zeros(y.shape).to(device)
            for i in range(self.transmission_length):
                # get the lsb of the state
                decoded_word[:, i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob, _ = acs_block(in_prob, priors[:, i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob
            return decoded_word
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")
