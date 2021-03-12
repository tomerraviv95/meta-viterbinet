from python_code.utils.trellis_utils import create_transition_table, acs_block
from torch.nn import functional as F
from typing import Dict
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 50


class META_VNETDetector(nn.Module):

    def __init__(self,
                 n_states: int,
                 transmission_lengths: Dict[str, int]):

        super(META_VNETDetector, self).__init__()
        self.transmission_lengths = transmission_lengths
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(device)

    def forward(self, y: torch.Tensor, phase: str, var: list) -> torch.Tensor:
        in_prob = torch.zeros([y.shape[0], self.n_states]).to(device)
        # compute priors based on input list of NN paramters
        x = y.reshape(-1, 1)
        x = F.linear(x, var[0], var[1])
        x = torch.sigmoid(x)
        x = F.linear(x, var[2], var[3])
        x = nn.functional.relu(x)
        x = F.linear(x, var[4], var[5])
        priors = x.reshape(y.shape[0], y.shape[1], self.n_states)

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
