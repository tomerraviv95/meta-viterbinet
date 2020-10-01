import math

import torch
import torch.nn as nn
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Link(nn.Module):
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
        self.new_mat_creation()

    def create_states_to_edges_matrix(self):
        self.states_to_edges = torch.cat([self.transition_table.reshape(1, -1) == i for i in range(self.n_states)],
                                         dim=0).float().to(device)

    def create_llrs_to_edges_matrix(self):
        input_bit_per_edge = np.tile([0, 1], self.n_states)
        modulated_input_per_edge = 1 - 2 * input_bit_per_edge.reshape(1, -1)
        self.llrs_to_edges = torch.Tensor(modulated_input_per_edge).to(device)

    def new_mat_creation(self):
        self.new_mat = torch.Tensor(1 - 2 * (
            np.unpackbits(np.arange(self.n_states).astype(np.uint8).reshape(-1, 1), axis=1)[:,
            -self.memory_length:]).astype(int))

    def compare_select(self, x: torch.Tensor) -> [torch.Tensor, torch.LongTensor]:
        """
        The compare-select operation return the maximum probabilities and the edges' indices of the chosen
        maximal values.
        :param x: LLRs matrix of size [batch_size,2*n_states] - two following indices in each row belong to two
        competing edges that enter the same state
        :return: the maximal llrs (from every pair), and the absolute edges' indices
        """
        reshaped_x = x.reshape(-1, self.n_states, 2)
        max_values, absolute_max_ind = torch.min(reshaped_x, 2)
        return max_values.reshape(-1), absolute_max_ind

    def forward(self, in_prob: torch.Tensor, llrs: torch.Tensor,
                h: torch.Tensor) -> [torch.Tensor, torch.LongTensor]:
        """
        Viterbi ACS block
        :param in_prob: last stage probabilities, [batch_size,n_states]
        :param llrs: edge probabilities, [batch_size,1]
        :return: current stage probabilities, [batch_size,n_states]
        """
        # calculate path metrics and branch metrics, per edge
        pm_mat = in_prob
        bm_mat = llrs
        trellis = (pm_mat + bm_mat)[self.transition_table.long()]

        # return ACS output
        return self.compare_select(trellis)
