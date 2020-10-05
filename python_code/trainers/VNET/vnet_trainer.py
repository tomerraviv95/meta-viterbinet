from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.utils.metrics import calculate_error_rates
from python_code.utils.trellis_utils import calculate_states
from python_code.trainers.trainer import Trainer
import numpy as np
import torch
import os

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

        return 'ViterbiNet' + channel_state

    def initialize_detector(self):
        """
        Loads the VNET detector
        """
        self.detector = VNETDetector(n_states=self.n_states,
                                     transmission_lengths=self.transmission_lengths)

    def load_weights(self, snr: float, gamma: float):
        """
        Loads detector's weights from checkpoint, if exists
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

    def gamma_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """
        ser_total = np.zeros(len(self.snr_range['val']))

        for snr_ind, snr in enumerate(self.snr_range['val']):
            self.load_weights(snr, gamma)
            ser_total[snr_ind] = self.single_eval(snr, gamma)

        return ser_total

    def select_batch(self, gt_states: torch.LongTensor, soft_estimation: torch.Tensor):
        """
        Select a batch from the input and output label
        :param gt_states:
        :param soft_estimation:
        :return:
        """
        rand_ind = torch.multinomial(torch.arange(gt_states.shape[0]).float(),
                                     self.train_minibatch_size).long().to(device)
        return gt_states[rand_ind], soft_estimation[rand_ind]

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_states(self.memory_length, transmitted_words)
        gt_states_batch, input_batch = self.select_batch(gt_states, soft_estimation.reshape(-1, self.n_states))
        loss = self.criterion(input=input_batch, target=gt_states_batch)
        return loss


if __name__ == '__main__':
    dec = VNETTrainer()
    dec.train()
    # dec.evaluate()
