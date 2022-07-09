import torch.nn as nn
import torch
from torch.nn import functional as F

from dir_definitions import DEVICE

INPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
N_CLASSES = 2
START_VALUE_PADDING = -100


# Directional recurrent neural network (many-to-one)
class MetaLSTMDetector(nn.Module):
    """
    This class implements an LSTM meta-detector
    """

    def __init__(self):
        super(MetaLSTMDetector, self).__init__()

    def forward(self, y: torch.Tensor, phase: str, var: list) -> torch.Tensor:
        batch_size, transmission_length = y.size(0), y.size(1)
        # accumulate h and c for backprop.
        h_n_seq_layers = {}
        c_n_seq_layers = {}
        for ind_layer in range(NUM_LAYERS):
            h_n_seq_layers[ind_layer] = [torch.zeros(batch_size, HIDDEN_SIZE).to(DEVICE)]
            c_n_seq_layers[ind_layer] = [torch.zeros(batch_size, HIDDEN_SIZE).to(DEVICE)]

        padded_y = torch.nn.functional.pad(y, [0, INPUT_SIZE - 1, 0, 0], value=START_VALUE_PADDING)
        sequence_y = torch.cat([torch.roll(padded_y.unsqueeze(1), i, 2) for i in range(INPUT_SIZE - 1, -1, -1)],
                               dim=1)
        sequence_y = sequence_y.transpose(1, 2)[:, :transmission_length]
        lstm_out = torch.zeros(batch_size, transmission_length, HIDDEN_SIZE).to(DEVICE)

        # pass sequence through model
        for ind_seq in range(transmission_length):
            x = sequence_y[:, ind_seq]  # batch_size, INPUT_SIZE # ind: t
            for ind_layer in range(NUM_LAYERS):
                # call latest h and c for current layer
                h_n = h_n_seq_layers[ind_layer][-1]
                c_n = c_n_seq_layers[ind_layer][-1]
                if ind_layer == 0:
                    input = x
                else:
                    input = h_n_seq_layers[ind_layer - 1][-1]

                mid_output = F.linear(h_n, var[4 * ind_layer + 1], var[4 * ind_layer + 3])
                i_f_g_o_curr_layer = F.linear(input, var[4 * ind_layer],
                                              var[4 * ind_layer + 2]) + mid_output  # batch_size, HIDDEN_SIZE*4
                i_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, :HIDDEN_SIZE])  # batch_size, HIDDEN_SIZE
                f_curr_layer = torch.sigmoid(
                    i_f_g_o_curr_layer[:, HIDDEN_SIZE:2 * HIDDEN_SIZE])  # batch_size, HIDDEN_SIZE
                g_curr_layer = torch.tanh(
                    i_f_g_o_curr_layer[:, 2 * HIDDEN_SIZE:3 * HIDDEN_SIZE])  # batch_size, HIDDEN_SIZE
                o_curr_layer = torch.sigmoid(
                    i_f_g_o_curr_layer[:, 3 * HIDDEN_SIZE:4 * HIDDEN_SIZE])  # batch_size, HIDDEN_SIZE
                c_n_seq_layers[ind_layer].append(f_curr_layer * c_n + i_curr_layer * g_curr_layer)
                h_n_seq_layers[ind_layer].append(o_curr_layer * torch.tanh(c_n_seq_layers[ind_layer][-1]))
            lstm_out[:, ind_seq] = h_n_seq_layers[NUM_LAYERS - 1][-1]
        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = F.linear((lstm_out.reshape(-1, HIDDEN_SIZE)), var[-2], var[-1]).reshape(batch_size,
                                                                                      transmission_length,
                                                                                      N_CLASSES)

        if phase == 'val':
            # Decode the output
            return torch.argmax(out, dim=2).float()
        else:
            return out
