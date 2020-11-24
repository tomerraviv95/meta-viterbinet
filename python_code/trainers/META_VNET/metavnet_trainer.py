from python_code.detectors.META_VNET.meta_vnet_detector import META_VNETDetector
from python_code.detectors.VNET.vnet_detector import VNETDetector
from python_code.utils.python_utils import copy_model
from python_code.utils.trellis_utils import calculate_states
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def initialize_meta_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.meta_detector = META_VNETDetector(n_states=self.n_states,
                                               transmission_lengths=self.transmission_lengths)

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
        copy_model(source_model=self.saved_detector, dest_model=self.detector)
        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            soft_estimation = self.detector(received_word, 'train')
            labels = detected_word if ser > 0 else encoded_word
            self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)

    def windowed_online_training(self, buffer_detected, buffer_received, buffer_encoded, count, detected_word,
                                 encoded_word,
                                 received_word, ser):
        copy_model(source_model=self.saved_detector, dest_model=self.detector)
        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            soft_estimation = self.detector(received_word, 'train')
            labels = detected_word if ser > 0 else encoded_word
            self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)
            for f in range(1, self.subframes_in_frame + 1):
                prev_ind = count - f
                if prev_ind >= 0:
                    # calculate soft values
                    soft_estimation = self.detector(buffer_received[prev_ind].reshape(1, -1), 'train')
                    labels = buffer_detected[prev_ind].reshape(1, -1) if ser > 0 else buffer_encoded[prev_ind].reshape(
                        1, -1)
                    self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)


if __name__ == '__main__':
    dec = METAVNETTrainer()
    # dec.meta_train()
    dec.evaluate()
