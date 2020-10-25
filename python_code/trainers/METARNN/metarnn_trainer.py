from typing import Tuple, Union

from python_code.detectors.METARNN.meta_rnn_detector import MetaRNNDetector
from python_code.detectors.RNN.rnn_detector import RNNDetector
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from python_code.trainers.trainer import Trainer
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SER_THRESH = 0.02
SELF_SUPERVISED_ITERATIONS = 500


class MetaRNNTrainer(Trainer):
    """
    Trainer for the RNN model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'Meta RNN' + channel_state

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = RNNDetector()
        self.meta_detector = MetaRNNDetector()

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        loss = self.criterion(input=soft_estimation.reshape(-1, 2), target=transmitted_words.long().reshape(-1))
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
    dec = MetaRNNTrainer()
    dec.meta_train()
    # dec.evaluate()
    # dec.count_parameters()
