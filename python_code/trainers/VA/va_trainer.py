from python_code.detectors.VA.va_detector import VADetector
from python_code.trainers.trainer import Trainer
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
                                   noisy_est_var=self.noisy_est_var)

    def train(self):
        raise NotImplementedError("No training implemented for this decoder!!!")


if __name__ == '__main__':
    dec = VATrainer()
    dec.evaluate()
