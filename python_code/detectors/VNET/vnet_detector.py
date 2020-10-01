from python_code.channel.channel_estimation import estimate_channel
from python_code.detectors.VNET.learnable_link import Link
import itertools
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UINT8_CONSTANT = 8
HIDDEN1_SIZE, HIDDEN2_SIZE = 100, 50


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    next state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


class VNETDetector(nn.Module):
    """
    This implements the VA decoder by unfolding into a neural network
    """

    def __init__(self,
                 n_states: int,
                 memory_length: int,
                 transmission_length: int,
                 channel_type: str,
                 noisy_est_var: float):

        super(VNETDetector, self).__init__()
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
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = []
        layers.append(nn.Linear(1, HIDDEN1_SIZE))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(HIDDEN2_SIZE, self.n_states))
        self.net = nn.Sequential(*layers).to(device)

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

    def run(self, y: torch.Tensor, h, phase):
        """
        The forward pass of the Viterbi algorithm
        :param y: input values (batch)
        """
        in_prob = torch.zeros(self.n_states).to(device)
        h_tensor = torch.Tensor(h).to(device)

        priors = self.net(y.reshape(-1, 1)).T

        if phase == 'val':
            ml_path_bits = torch.zeros(y.shape).to(device)

            for i in range(self.transmission_length):
                ml_path_bits[:, i] = torch.argmin(in_prob) % 2
                out_prob, inds = self.basic_layer(in_prob, -priors[:, i], h_tensor)
                # update in-probabilities for next layer, clipping above and below thresholds
                in_prob = out_prob

            return ml_path_bits
        else:
            return priors
