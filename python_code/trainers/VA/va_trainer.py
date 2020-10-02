from python_code.detectors.VA.va_detector import VADetector
from python_code.utils.metrics import calculate_error_rates
from python_code.trainers.trainer import Trainer
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VATrainer(Trainer):
    """
    Trainer for the VA model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'Viterbi' + channel_state

    def initialize_detector(self):
        """
        Loads the VA detector
        """
        self.detector = VADetector(n_states=self.n_states,
                                   memory_length=self.memory_length,
                                   transmission_length=self.transmission_length,
                                   channel_type=self.channel_type,
                                   channel_blocks=self.channel_blocks,
                                   noisy_est_var=self.noisy_est_var)

    def single_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """
        ser_total = np.zeros(len(self.snr_range['val']))

        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(
            snr_list=self.snr_range['val'],
            gamma=gamma)

        # decode and calculate accuracy
        decoded_words = self.detector(received_words, 'val', gamma)

        for snr_ind in range(len(self.snr_range['val'])):
            start_ind = snr_ind * self.val_words
            end_ind = (snr_ind + 1) * self.val_words
            ser, fer, err_indices = calculate_error_rates(decoded_words[start_ind:end_ind],
                                                          transmitted_words[start_ind:end_ind])
            ser_total[snr_ind] = ser

        return ser_total

    def train(self):
        raise NotImplementedError("No training implemented for this decoder!!!")


if __name__ == '__main__':
    dec = VATrainer()
    dec.evaluate()
