from python_code.detectors.LSTM.lstm_detector import LSTMDetector
from python_code.trainers.trainer import Trainer
import torch


class LSTMTrainer(Trainer):
    """
    Trainer for the LSTM model.
    """

    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.noisy_est_var > 0:
            channel_state = ', CSI uncertainty'
        else:
            channel_state = ', perfect CSI'

        return 'LSTM' + channel_state

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = LSTMDetector()

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        gt_batch, input_batch = self.select_batch(transmitted_words.long().reshape(-1),
                                                  soft_estimation.reshape(-1, 2))
        loss = self.criterion(input=input_batch, target=gt_batch)
        return loss

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - train on the detected/re-encoded word only if the ser is below some threshold.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        """
        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            soft_estimation = self.detector(rx, 'train')
            self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=tx)


if __name__ == '__main__':
    dec = LSTMTrainer()
    dec.train()
    # dec.evaluate()
