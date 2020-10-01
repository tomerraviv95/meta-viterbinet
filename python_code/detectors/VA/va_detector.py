from python_code.channel.channel_estimation import estimate_channel
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.VA.link import Link
import itertools
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UINT8_CONSTANT = 8


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    next state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


class VADetector(nn.Module):
    """
    This implements the VA decoder by unfolding into a neural network
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_length: int,
                 channel_type: str,
                 noisy_est_var: float):

        super(VADetector, self).__init__()
        self.start_state = 0
        self.memory_length = memory_length
        self.transmission_length = transmission_length
        self.n_states = n_states
        self.channel_type = channel_type
        self.noisy_est_var = noisy_est_var
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)
        # initialize all stages of the cva detectors
        self.init_layers()

    def init_layers(self):
        self.basic_layer = Link(n_states=self.n_states,
                                memory_length=self.memory_length,
                                transition_table_array=self.transition_table_array).to(device)

    def forward(self, y: torch.Tensor, phase: str, snr: float, gamma: float) -> torch.Tensor:
        """
        The circular Viterbi algorithm
        :param y: input llrs (batch)
        :param phase: 'val' or 'train'
        :return: batch of decoded binary words
        """
        # channel_estimate
        h = estimate_channel(self.memory_length, gamma, noisy_est_var=self.noisy_est_var)
        # forward pass
        estimated_word = self.run(y, h, phase)
        return estimated_word

    def run(self, y: torch.Tensor, h: np.ndarray, phase: str):
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        """
        self.batch_size = y.size(0)
        in_prob = torch.zeros(self.n_states).to(device)
        h_tensor = torch.Tensor(h).to(device)

        self.new_mat = torch.Tensor(1 - 2 * (
            np.unpackbits(np.arange(self.n_states).astype(np.uint8).reshape(-1, 1), axis=1)[:,
            -self.memory_length:]).astype(int)).to(device)
        isi_per_state = torch.mm(self.new_mat, h_tensor.T)
        priors = torch.abs(y - isi_per_state)
        ml_path_bits = torch.zeros(y.shape).to(device)

        for i in range(self.transmission_length):
            ml_path_bits[:, i] = torch.argmin(in_prob) % 2
            out_prob, inds = self.basic_layer(in_prob, priors[:, i], h_tensor)
            # update in-probabilities for next layer, clipping above and below thresholds
            in_prob = out_prob

        if phase == 'val':
            return ml_path_bits
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")
