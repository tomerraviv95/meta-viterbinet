from python_code.utils.trellis_utils import create_transition_table, acs_block
from typing import Dict
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 50


class VNETDetector(nn.Module):
    """
    This implements the VA decoder by an NN on each stage
    """

    def __init__(self,
                 n_states: int,
                 transmission_lengths: Dict[str, int]):

        super(VNETDetector, self).__init__()
        self.transmission_lengths = transmission_lengths
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE),
                  nn.Sigmoid(),
                  nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN2_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None,
                count: int = None) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :returns if in 'train' - the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        # initialize input probabilities
        in_prob = torch.zeros([y.shape[0], self.n_states]).to(device)
        # compute priors
        priors = self.net(y.reshape(-1, 1)).reshape(y.shape[0], y.shape[1], self.n_states)

        if phase == 'val':
            decoded_word = torch.zeros(y.shape).to(device)
            for i in range(self.transmission_lengths['val']):
                # get the lsb of the state
                decoded_word[:, i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob, _ = acs_block(in_prob, -priors[:, i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return decoded_word
        else:
            return priors
