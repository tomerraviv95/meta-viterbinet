import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 2
N_CLASSES = 2
START_VALUE_PADDING = -100


# Directional recurrent neural network (many-to-one)
class LSTMDetector(nn.Module):
    """
    This class implements an LSTM detector
    """

    def __init__(self):
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, bidirectional=False).to(device)
        self.fc = nn.Linear(HIDDEN_SIZE, N_CLASSES).to(device)

    def forward(self, y: torch.Tensor, phase: str, snr: float = None, gamma: float = None) -> torch.Tensor:
        """
        The forward pass of the LSTM detector
        :param y: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :param snr: channel snr
        :param gamma: channel coefficient
        :return: if in 'train' - the estimated bitwise prob [batch_size,transmission_length,N_CLASSES]
        if in 'val' - the detected words [n_batch,transmission_length]
        """
        batch_size, transmission_length = y.size(0), y.size(1)

        # Set initial states
        h_n = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE).to(device)
        c_n = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE).to(device)

        # pad and reshape y to the proper shape - (batch_size,seq_length,input_size)
        padded_y = torch.nn.functional.pad(y, [0, INPUT_SIZE - 1, 0, 0], value=START_VALUE_PADDING)
        sequence_y = torch.cat([torch.roll(padded_y.unsqueeze(1), i, 2) for i in range(INPUT_SIZE - 1, -1, -1)], dim=1)
        sequence_y = sequence_y.transpose(1, 2)[:, :transmission_length]

        # Forward propagate LSTM - lstm_out: tensor of shape (batch_size, seq_length, hidden_size*2)
        lstm_out = torch.zeros(batch_size, transmission_length, HIDDEN_SIZE).to(device)
        for i in range(batch_size):
            lstm_out[i], _ = self.lstm(sequence_y[i].unsqueeze(0),
                                       (h_n[:, i].unsqueeze(1).contiguous(), c_n[:, i].unsqueeze(1).contiguous()))

        # out: tensor of shape (batch_size, seq_length, N_CLASSES)
        out = self.fc(lstm_out.reshape(-1, HIDDEN_SIZE)).reshape(batch_size, transmission_length, N_CLASSES)

        if phase == 'val':
            # Decode the output
            return torch.argmax(out, dim=2).float()
        else:
            return out
