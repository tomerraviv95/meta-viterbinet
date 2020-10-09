from python_code.utils.trellis_utils import create_transition_table, acs_block
from typing import Dict
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
N_CLASSES = 2


# Bidirectional recurrent neural network (many-to-one)
class RNNDetector(nn.Module):
    """
    This implements a sliding RNN detector
    """

    def __init__(self):
        super(RNNDetector, self).__init__()
        self.initialize_rnn()

    def initialize_rnn(self):
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, bidirectional=True).to(device)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, N_CLASSES).to(device)  # 2 for bidirection

    def forward(self, y: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the RNN detector
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        batch_size, transmission_length = y.size(0), y.size(1)
        # Set initial states
        h_n = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)  # 2 for bidirection
        c_n = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        # pad and reshape y to the proper shape
        padded_y = torch.nn.functional.pad(y, [0, INPUT_SIZE - 1, 0, 0],value=-100)
        sequence_y = torch.cat([torch.roll(padded_y.unsqueeze(1), i, 2) for i in range(INPUT_SIZE - 1, -1, -1)], dim=1)
        sequence_y = sequence_y.transpose(1, 2)[:, :transmission_length]
        # Forward propagate LSTM - lstm_out: tensor of shape (batch_size, seq_length, hidden_size*2)
        lstm_out, _ = self.lstm(sequence_y, (h_n, c_n))
        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = self.fc(lstm_out.reshape(-1, HIDDEN_SIZE * 2)).reshape(batch_size, transmission_length, N_CLASSES)

        if phase == 'val':
            # Decode the output
            return torch.argmax(out, dim=2)
        else:
            return out
