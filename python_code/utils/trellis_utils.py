import torch
import numpy as np

from dir_definitions import DEVICE


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    previous state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(n_states, 2)
    return transition_table


def acs_block(in_prob: torch.Tensor, llrs: torch.Tensor, transition_table: torch.Tensor, n_states: int) -> [
    torch.Tensor, torch.LongTensor]:
    """
    Viterbi ACS block
    :param in_prob: last stage probabilities, [batch_size,n_states]
    :param llrs: edge probabilities, [batch_size,1]
    :param transition_table: transitions
    :param n_states: number of states
    :return: current stage probabilities, [batch_size,n_states]
    """
    transition_ind = transition_table.reshape(-1).repeat(in_prob.size(0)).long()
    batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * n_states)
    trellis = (in_prob + llrs)[batches_ind, transition_ind]
    reshaped_trellis = trellis.reshape(-1, n_states, 2)
    return torch.min(reshaped_trellis, dim=2)


def calculate_states(memory_length: int, transmitted_words: torch.Tensor) -> torch.Tensor:
    """
    calculates the state for the transmitted words
    :param memory_length: length of channel memory
    :param transmitted_words: channel transmitted words
    :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
    """
    padded = torch.cat([transmitted_words, torch.zeros([transmitted_words.shape[0], memory_length]).to(DEVICE)], dim=1)
    unsqueezed_padded = padded.unsqueeze(dim=1)
    blockwise_words = torch.cat([unsqueezed_padded[:, :, i:-memory_length + i] for i in range(memory_length)], dim=1)
    states_enumerator = (2 ** torch.arange(memory_length)).reshape(1, -1).float().to(DEVICE)
    gt_states = torch.sum(blockwise_words.transpose(1, 2).reshape(-1, memory_length) * states_enumerator,
                          dim=1).long()
    return gt_states
