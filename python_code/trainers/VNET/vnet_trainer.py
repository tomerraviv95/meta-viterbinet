import os

from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.trainers.trainer import Trainer
import torch

from python_code.utils.trellis_utils import calculate_states

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

    def initialize_detector(self):
        """
        Loads the VNET detector
        """
        self.detector = VNETDetector(n_states=self.n_states,
                                     memory_length=self.memory_length,
                                     transmission_length=self.transmission_length,
                                     channel_type=self.channel_type,
                                     noisy_est_var=self.noisy_est_var)

    def load_weights(self, snr, gamma):
        """
        Loads detector's weights from highest checkpoint in run_name
        """
        if os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'):
            print(f'loading model from snr {snr} and gamma {gamma}')
            checkpoint = torch.load(os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'))
            try:
                self.detector.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'No checkpoint for snr {snr} and gamma {gamma} in run "{self.run_name}", starting from scratch')

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor) -> torch.Tensor:
        padded_b = torch.cat(
            [transmitted_words, torch.zeros([transmitted_words.shape[0], self.memory_length]).to(device)], dim=1)
        before = torch.cat([padded_b[:, i:-self.memory_length + i] for i in range(self.memory_length)], dim=0)
        new_h = (2 ** torch.arange(self.memory_length)).float().reshape(1, -1).to(device)
        gt_states = torch.mm(new_h, before).reshape(-1).long()
        input = soft_estimation.T
        rand_ind = torch.multinomial(torch.arange(transmitted_words.shape[1]).float(),
                                     self.train_minibatch_size).long().to(device)
        loss = self.criterion(input=input[rand_ind], target=gt_states[rand_ind])
        return loss


if __name__ == '__main__':
    dec = VNETTrainer()
    dec.train()
    # dec.evaluate()
