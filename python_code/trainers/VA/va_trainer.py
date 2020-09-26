from python_code.detectors.VA.va_detector import VADetector
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VATrainer(Trainer):
    """
    Trainer for the VA model.
    """

    def __init__(self, config_path=None, **kwargs):
        self.n_states = None
        super().__init__(config_path, **kwargs)

    def __name__(self):
        return f'VA'

    def load_detector(self):
        """
        Loads the VA decoder
        """
        self.decoder = VADetector(n_states=self.n_states,
                                  transmission_length=self.transmission_length)

    def train(self):
        raise NotImplementedError("No training implemented for this decoder!!!")


if __name__ == '__main__':
    dec = VATrainer()
    dec.evaluate()
