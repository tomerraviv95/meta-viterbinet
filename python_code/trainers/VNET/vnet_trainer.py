from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.trainers.trainer import Trainer
import torch

from python_code.utils.trellis_utils import calculate_starting_state_for_tbcc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VNETTrainer(Trainer):
    """
    Trainer for the VNET model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'VNET' + channel_state

    def load_detector(self):
        """
        Loads the VNET decoder
        """
        self.decoder = VNETDetector(n_states=self.n_states,
                                    memory_length=self.memory_length,
                                    transmission_length=self.transmission_length,
                                    channel_type=self.channel_type,
                                    noisy_est_var=self.noisy_est_var)

        if self.load_from_checkpoint:
            self.load_last_checkpoint()

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor) -> torch.Tensor:
        labels = calculate_starting_state_for_tbcc(self.n_states, transmitted_words)
        loss = self.criterion(input=soft_estimation[:, :-self.memory_length].reshape(-1, self.n_states),
                              target=labels.reshape(-1))
        return loss


if __name__ == '__main__':
    dec = VNETTrainer()
    dec.train()
