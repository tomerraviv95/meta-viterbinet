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
    all_states = list(itertools.product([0, 1], repeat=int(np.log2(n_states))))
    repeated_all_states = np.repeat(np.array(all_states), 2, axis=0)
    # create new states by inserting 0 or 1 for each state
    inserted_bits = np.tile(np.array([0, 1]), n_states).reshape(-1, 1)
    mapped_states = np.concatenate(
        [np.zeros([2 * n_states, UINT8_CONSTANT - repeated_all_states.shape[1]]), inserted_bits,
         repeated_all_states[:, :-1]],
        axis=1)
    numbered_mapped_states = np.packbits(mapped_states.astype(np.uint8), axis=1)
    transition_table = numbered_mapped_states.reshape(n_states, 2)
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
        h = estimate_channel(self.memory_length, snr, gamma, noisy_est_var=self.noisy_est_var)
        # forward pass
        self.run(y, h)
        # trace-back
        estimated_word = self.traceback(phase)
        return estimated_word

    def run(self, y: torch.Tensor, h: np.ndarray):
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        """
        self.batch_size = y.size(0)

        # set initial probabilities
        self.initial_in_prob = torch.zeros((self.batch_size, self.n_states)).to(device)

        previous_states = torch.zeros(
            [self.batch_size, self.n_states, self.transmission_length]).to(device)
        out_prob_mat = torch.zeros(
            [self.batch_size, self.n_states, self.transmission_length]).to(device)
        in_prob = self.initial_in_prob.clone()
        previous_symbols_per_state = torch.zeros([self.batch_size, self.memory_length - 1, self.n_states]).to(device)
        h_tensor = torch.Tensor(h).to(device)

        for i in range(self.transmission_length):
            out_prob, inds = self.basic_layer(in_prob, y[:, i], previous_symbols_per_state, h_tensor)
            # update the previous state (each index corresponds to the state out of the total n_states)
            previous_states[:, :, i] = self.transition_table[
                torch.arange(self.n_states).repeat(self.batch_size, 1), inds]
            out_prob_mat[:, :, i] = out_prob
            # update in-probabilities for next layer, clipping above and below thresholds
            in_prob = out_prob
            previous_symbols_per_state = self.update_previous_symbols_per_state(inds, previous_symbols_per_state)

        self.previous_states = previous_states

    def update_previous_symbols_per_state(self, inds: torch.Tensor,
                                          previous_symbols_per_state: torch.Tensor) -> torch.Tensor:
        for j in range(self.memory_length - 2, 0, -1):
            previous_symbols_per_state[:, j] = previous_symbols_per_state[:, j - 1]
        if self.channel_type == 'ISI_AWGN':
            previous_symbols_per_state[:, 0] = BPSKModulator.modulate(inds)
        else:
            raise Exception("No such channel exists!!!")
        return previous_symbols_per_state

    def traceback(self, phase: str) -> torch.Tensor:
        """
        Trace-back of the VA
        :return: binary decoded codewords
        """
        if phase == 'val':
            # trace back unit
            most_likely_state = self.start_state
            ml_path_bits = torch.zeros([self.batch_size, self.transmission_length]).to(device)

            # traceback - loop on all stages, from last to first, saving the most likely path
            for i in range(self.transmission_length - 1, -1, -1):
                most_likely_state = self.previous_states[torch.arange(self.batch_size), most_likely_state, i].long()
                ml_path_bits[:, i] = (most_likely_state >= self.n_states // 2)
            return ml_path_bits
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")
