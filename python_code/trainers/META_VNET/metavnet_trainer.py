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

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - train on the detected/re-encoded word only if the ser is below some threshold.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        """
        copy_model(source_model=self.saved_detector, dest_model=self.detector)
        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            soft_estimation = self.detector(rx, 'train')
            self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=tx)


if __name__ == '__main__':
    dec = METAVNETTrainer()
    # dec.meta_train()
    dec.evaluate()
