from python_code.detectors.META_LSTM.meta_lstm_detector import MetaLSTMDetector
from python_code.detectors.LSTM.lstm_detector import LSTMDetector
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaLSTMTrainer(Trainer):
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

        return 'Meta LSTM' + channel_state

    def initialize_detector(self):
        """
        Loads the ViterbiNet detector
        """
        self.detector = LSTMDetector()

    def initialize_meta_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.meta_detector = MetaLSTMDetector()

    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param soft_estimation: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        loss = self.criterion(input=soft_estimation.reshape(-1, 2), target=transmitted_words.long().reshape(-1))
        return loss

    def online_training(self, detected_word: torch.Tensor, encoded_word: torch.Tensor, received_word: torch.Tensor,
                        ser: float):
        """
        Online training module - train on the detected/re-encoded word only if the ser is below some threshold.
        Start from the saved meta-trained weights.
        :param detected_word: detected channel codeword
        :param encoded_word: re-encoded decoded word
        :param received_word: the channel received word
        :param ser: calculated ser for the word
        """
        self.detector = self.saved_detector.copy()
        if ser <= self.ser_thresh:
            # run training loops
            for i in range(self.self_supervised_iterations):
                # calculate soft values
                soft_estimation = self.detector(received_word, 'train')
                labels = detected_word if ser > 0 else encoded_word
                self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)


if __name__ == '__main__':
    dec = MetaLSTMTrainer()
    dec.meta_train()
    # dec.evaluate()
    # dec.count_parameters()
