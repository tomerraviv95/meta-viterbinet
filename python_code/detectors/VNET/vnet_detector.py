from python_code.utils.trellis_utils import create_transition_table, acs_block
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
                 transmission_length: int):

        super(VNETDetector, self).__init__()
        self.start_state = 0
        self.transmission_length = transmission_length
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

    def forward(self, y: torch.Tensor, phase: str, *args) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param y: input values
        :param phase: 'train' or 'val'
        """
        # initialize input probabilities
        in_prob = torch.zeros([y.shape[0], self.n_states]).to(device)
        # compute priors
        priors = self.net(y.reshape(-1, 1)).reshape(y.shape[0], y.shape[1], self.n_states)

        if phase == 'val':
            decoded_word = torch.zeros(y.shape).to(device)
            for i in range(self.transmission_length):
                # get the lsb of the state
                decoded_word[:, i] = torch.argmin(in_prob, dim=1) % 2
                # run one Viterbi stage
                out_prob, _ = acs_block(in_prob, -priors[:, i], self.transition_table, self.n_states)
                # update in-probabilities for next layer
                in_prob = out_prob

            return decoded_word
        else:
            return priors
