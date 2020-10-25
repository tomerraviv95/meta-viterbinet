from typing import Union
from python_code.detectors.METAVNET.meta_vnet_detector import META_VNETDetector
from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from python_code.utils.trellis_utils import calculate_states
from python_code.trainers.trainer import Trainer
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SER_THRESH = 0.02
SELF_SUPERVISED_ITERATIONS = 500


class METAVNETTrainer(Trainer):
    """
    Trainer for the ViterbiNet model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'MetaViterbiNet' + channel_state

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = VNETDetector(n_states=self.n_states,
                                     transmission_lengths=self.transmission_lengths)
        self.meta_detector = META_VNETDetector(n_states=self.n_states,
                                               transmission_lengths=self.transmission_lengths)

    def load_weights(self, snr: float, gamma: float):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists
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

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_states(self.memory_length, transmitted_words)
        loss = self.criterion(input=soft_estimation.reshape(-1, self.n_states), target=gt_states)
        return loss

    def online_training(self, detected_word, encoded_word, gamma, received_word, ser, snr):
        self.load_weights(snr, gamma)
        self.deep_learning_setup()
        if ser <= SER_THRESH:
            # run training loops
            for i in range(SELF_SUPERVISED_ITERATIONS):
                # calculate soft values
                soft_estimation = self.detector(received_word, 'train')
                labels = detected_word if ser > 0 else encoded_word
                self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)


if __name__ == '__main__':
    dec = METAVNETTrainer()
    # dec.meta_train()
    dec.evaluate()
