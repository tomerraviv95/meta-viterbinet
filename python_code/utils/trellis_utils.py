import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_states(n_states: int, transmitted_words: torch.Tensor):
    """
    calculates the starting state for the give words (u_det is information + crc word)
    take last bits, and pass through code's trellis
    :param u_det: size [batch_size,info_length+crc_length]
    :return: vector of length batch_size with values in the range of 0,1,...,n_states-1
    """
    all_states = torch.zeros([transmitted_words.size(0), transmitted_words.size(1)+1]).long().to(device)
    for i in range(transmitted_words.size(1)):
        all_states[:, i + 1] = map_bit_and_state_to_next_state(n_states,
                                                               transmitted_words[:, i],
                                                               all_states[:, i])
    return all_states[:, 1:]


def map_bit_and_state_to_next_state(n_states: int, bit: torch.Tensor, state: torch.Tensor):
    """
    Based on the current srs and bits arrays. For example:
    and so on...
    """
    next_state = torch.zeros_like(state)
    zeros_ind = (bit == 0)
    next_state[zeros_ind] = state[zeros_ind] // 2
    ones_ind = (bit == 1)
    next_state[ones_ind] = (state[ones_ind] // 2 + (n_states / 2)).long()
    return next_state.long()
