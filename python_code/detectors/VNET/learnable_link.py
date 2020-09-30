import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN1_SIZE, HIDDEN2_SIZE = 100, 50


class LearnableLink(nn.Module):
    def __init__(self, n_states: int,
                 memory_length: int,
                 transition_table_array: np.ndarray):
        self.n_states = n_states
        self.transition_table_array = transition_table_array
        self.transition_table = torch.Tensor(transition_table_array)
        self.memory_length = memory_length
        super().__init__()

        # create matrices
        self.create_states_to_edges_matrix()
        self.create_llrs_to_edges_matrix()
        self.initialize_dnn()

    def initialize_dnn(self):
        self.fc1 = nn.Linear(1, HIDDEN1_SIZE)  # 6*6 from image dimension
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc3 = nn.Linear(HIDDEN2_SIZE, self.n_states)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def create_states_to_edges_matrix(self):
        self.states_to_edges = torch.cat([self.transition_table.reshape(1, -1) == i for i in range(self.n_states)],
                                         dim=0).float().to(device)

    def create_llrs_to_edges_matrix(self):
        input_bit_per_edge = np.tile([0, 1], self.n_states)
        modulated_input_per_edge = 1 - 2 * input_bit_per_edge.reshape(1, -1)
        self.llrs_to_edges = torch.Tensor(modulated_input_per_edge).to(device)

    def create_last_bits_mat(self):
        binary_bits_mat = np.unpackbits(np.repeat(np.arange(self.n_states), 2).astype(np.uint8).reshape(-1, 1), axis=1)
        only_memory_length_last_bits = binary_bits_mat[:, -self.memory_length:-1]
        return only_memory_length_last_bits

    def compare_select(self, x: torch.Tensor) -> [torch.Tensor, torch.LongTensor]:
        """
        The compare-select operation return the maximum probabilities and the edges' indices of the chosen
        maximal values.
        :param x: LLRs matrix of size [batch_size,2*n_states] - two following indices in each row belong to two
        competing edges that enter the same state
        :return: the maximal llrs (from every pair), and the absolute edges' indices
        """
        reshaped_x = x.reshape(-1, self.n_states, 2)
        max_values, absolute_max_ind = torch.max(reshaped_x, 2)
        return max_values, absolute_max_ind

    def dnn(self, llrs: torch.Tensor) -> torch.Tensor:
        out1 = self.fc1(llrs.reshape(-1, 1))
        act_out1 = self.sigmoid(out1)
        out2 = self.fc2(act_out1)
        act_out2 = self.relu(out2)
        out3 = self.fc3(act_out2)
        return out3

    def forward(self, in_prob: torch.Tensor, llrs: torch.Tensor, marginal_costs_mat: torch.Tensor, i: int) -> [
        torch.Tensor, torch.LongTensor]:
        """
        Viterbi ACS block
        :param in_prob: last stage probabilities, [batch_size,n_states]
        :param llrs: edge probabilities, [batch_size,1]
        :return: current stage probabilities, [batch_size,n_states]
        """
        # calculate path metrics and branch metrics, per edge
        pm_mat = torch.mm(in_prob, self.states_to_edges)
        marginal_costs = self.dnn(llrs)
        marginal_costs_mat[:, i, :] = marginal_costs
        bm_mat = torch.mm(marginal_costs, self.states_to_edges)

        # return ACS output
        link_output = self.compare_select(pm_mat + bm_mat)
        return link_output
