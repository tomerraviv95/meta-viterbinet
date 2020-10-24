import torch.nn as nn
import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
N_CLASSES = 2


# Directional recurrent neural network (many-to-one)
class META_RNNDetector(nn.Module):
    """
    This class implements a sliding RNN detector
    """
    def __init__(self):
        super(META_RNNDetector, self).__init__()
        self.initialize_rnn()

    def initialize_rnn(self):
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS

    def forward(self, y: torch.Tensor, phase: str, var: list) -> torch.Tensor:
        # 0~3: layer 1
        # var[0]: for x, weight
        # var[1]: for h, weight
        # var[2]: for x, bias
        # var[3]: for h, bias
        # 4~7: layer 2
        # var[4]: for x, weight
        # var[5]: for h, weight
        # var[6]: for x, bias
        # var[7]: for h, bias
        # fc
        # var[8]: fc weight
        # var[9]: fc bias
        with torch.autograd.set_detect_anomaly(True):
            batch_size, transmission_length = y.size(0), y.size(1)
            # h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            # c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            # accumulate h and c for backprop.
            h_n_seq_layers = {}
            c_n_seq_layers = {}
            for ind_layer in range(self.num_layers):
                h_n_seq_layers[ind_layer] = [torch.zeros(batch_size, self.hidden_size).to(device)]
                c_n_seq_layers[ind_layer] = [torch.zeros(batch_size, self.hidden_size).to(device)]

            padded_y = torch.nn.functional.pad(y, [0, INPUT_SIZE - 1, 0, 0], value=-100)
            sequence_y = torch.cat([torch.roll(padded_y.unsqueeze(1), i, 2) for i in range(INPUT_SIZE - 1, -1, -1)],
                                   dim=1)
            sequence_y = sequence_y.transpose(1, 2)[:, :transmission_length]
            lstm_out = torch.zeros(batch_size, transmission_length, HIDDEN_SIZE).to(device)

            for ind_seq in range(transmission_length):
                x = sequence_y[:, ind_seq]  # batch_size, INPUT_SIZE # ind: t
                for ind_layer in range(self.num_layers):
                    # call latest h and c for current layer
                    h_n = h_n_seq_layers[ind_layer][-1]
                    c_n = c_n_seq_layers[ind_layer][-1]
                    if ind_layer == 0:
                        input = x
                    else:
                        input = h_n_seq_layers[ind_layer - 1][-1]
                    i_f_g_o_curr_layer = F.linear(input, var[4 * ind_layer], var[4 * ind_layer + 2]) + F.linear(h_n,
                                                                                                                var[
                                                                                                                    4 * ind_layer + 1],
                                                                                                                var[
                                                                                                                    4 * ind_layer + 3])  # batch_size, HIDDEN_SIZE*4
                    i_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, :HIDDEN_SIZE])  # batch_size, HIDDEN_SIZE
                    f_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, HIDDEN_SIZE:2 * HIDDEN_SIZE]) # batch_size, HIDDEN_SIZE
                    g_curr_layer = torch.tanh(i_f_g_o_curr_layer[:, 2 * HIDDEN_SIZE:3 * HIDDEN_SIZE]) # batch_size, HIDDEN_SIZE
                    o_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, 3 * HIDDEN_SIZE:4 * HIDDEN_SIZE]) # batch_size, HIDDEN_SIZE
                    c_n_seq_layers[ind_layer].append(f_curr_layer * c_n + i_curr_layer * g_curr_layer)
                    h_n_seq_layers[ind_layer].append(o_curr_layer * torch.tanh(c_n_seq_layers[ind_layer][-1]))
                lstm_out[:, ind_seq] = h_n_seq_layers[self.num_layers - 1][-1]
            # out: tensor of shape (batch_size, seq_length, N_CLASSES)
            out = F.linear((lstm_out.reshape(-1, HIDDEN_SIZE)), var[-2], var[-1]).reshape(batch_size,
                                                                                          transmission_length,
                                                                                          N_CLASSES)

            if phase == 'val':
                # Decode the output
                return torch.argmax(out, dim=2)
            else:
                return out





