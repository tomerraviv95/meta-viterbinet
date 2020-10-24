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
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, bidirectional=False).to(device)
        self.fc = nn.Linear(HIDDEN_SIZE, N_CLASSES).to(device)

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

        batch_size, transmission_length = y.size(0), y.size(1)
        h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_n = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        padded_y = torch.nn.functional.pad(y, [0, INPUT_SIZE - 1, 0, 0], value=-100)
        sequence_y = torch.cat([torch.roll(padded_y.unsqueeze(1), i, 2) for i in range(INPUT_SIZE - 1, -1, -1)], dim=1)
        sequence_y = sequence_y.transpose(1, 2)[:, :transmission_length]

        lstm_out = torch.zeros(batch_size, transmission_length, HIDDEN_SIZE).to(device)

        for ind_seq in range(transmission_length):
            x = sequence_y[:, ind_seq] # batch_size, INPUT_SIZE # ind: t
            print('x', x.shape)

            # i_f_g_o_curr_layer: batch_size, HIDDEN_SIZE*4


            for ind_layer in range(self.num_layers):
                if ind_layer == 0:
                    input = x
                else:
                    input = h_n[ind_layer-1]
                i_f_g_o_curr_layer = F.linear(input, var[4*ind_layer], var[4*ind_layer+2]) + F.linear(h_n[ind_layer], var[4*ind_layer+1], var[4*ind_layer+3])
                i_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, :HIDDEN_SIZE])
                f_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, HIDDEN_SIZE:2 * HIDDEN_SIZE])
                g_curr_layer = F.tanh(i_f_g_o_curr_layer[:, 2 * HIDDEN_SIZE:3 * HIDDEN_SIZE])
                o_curr_layer = torch.sigmoid(i_f_g_o_curr_layer[:, 3 * HIDDEN_SIZE:4 * HIDDEN_SIZE])
                c_n[ind_layer]  = f_curr_layer * c_n[ind_layer] + i_curr_layer * g_curr_layer  # update c_n
                h_n[ind_layer] = o_curr_layer * F.tanh(c_n[ind_layer])

                h_n_layer_1 = h_n[0] # batch_size, HIDDEN_SIZE # ind: t-1
                h_n_layer_2 = h_n[1]
                c_n_layer_1 = c_n[0] # batch_size, HIDDEN_SIZE
                c_n_layer_2 = c_n[1]

                i_f_g_o_layer_1 = F.linear(x, var[0], var[2]) + F.linear(h_n_layer_1, var[1], var[3]) # batch_size, HIDDEN_SIZE*4
                i_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, :HIDDEN_SIZE])
                f_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, HIDDEN_SIZE:2*HIDDEN_SIZE])
                g_layer_1 = F.tanh(i_f_g_o_layer_1[:, 2*HIDDEN_SIZE:3*HIDDEN_SIZE])
                o_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, 3*HIDDEN_SIZE:4*HIDDEN_SIZE])
                c_n_layer_1_updated = f_layer_1 * c_n_layer_1 + i_layer_1 * g_layer_1 # ind: t
                h_n_layer_1_updated = o_layer_1 * F.tanh(c_n_layer_1_updated)

                print('before', h_n[0])
                c_n[0] = c_n_layer_1_updated
                h_n[0] = h_n_layer_1_updated
                print('after', h_n[0])

                i_f_g_o_layer_2 = F.linear(h_n_layer_1_updated, var[4], var[6]) + F.linear(h_n_layer_2, var[5],
                                                                         var[7])  # batch_size, HIDDEN_SIZE*4
                i_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, :HIDDEN_SIZE])
                f_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, HIDDEN_SIZE:2 * HIDDEN_SIZE])
                g_layer_1 = F.tanh(i_f_g_o_layer_1[:, 2 * HIDDEN_SIZE:3 * HIDDEN_SIZE])
                o_layer_1 = torch.sigmoid(i_f_g_o_layer_1[:, 3 * HIDDEN_SIZE:4 * HIDDEN_SIZE])
                c_n_layer_1_updated = f_layer_1 * c_n_layer_1 + i_layer_1 * g_layer_1  # ind: t
                h_n_layer_1_updated = o_layer_1 * F.tanh(c_n_layer_1_updated)

                print('before', h_n[0])
                c_n[0] = c_n_layer_1_updated
                h_n[0] = h_n_layer_1_updated
                print('after', h_n[0])











        for i in range(batch_size):
            for ind_seq in range(transmission_length):
                x = sequence_y[i][ind_seq]
                i_f_g_o = F.linear(x, var[0], var[1])
                lstm_out[i, ind_seq] =
                h_n =
                c_n =



            lstm_out[i], _ = self.lstm(sequence_y[i].unsqueeze(0),
                                       (h_n[:, i].unsqueeze(1).contiguous(), c_n[:, i].unsqueeze(1).contiguous()))

        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = self.fc(lstm_out.reshape(-1, HIDDEN_SIZE)).reshape(batch_size, transmission_length, N_CLASSES)

        if phase == 'val':
            # Decode the output
            return torch.argmax(out, dim=2)
        else:
            return out
