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

        return 'ViterbiNet' + channel_state

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
        gt_states_batch, input_batch = self.select_batch(gt_states, soft_estimation.reshape(-1, self.n_states))
        loss = self.criterion(input=input_batch, target=gt_states_batch)
        return loss

    def single_eval(self, snr: float, gamma: float) -> float:
        """
        ViterbiNet single eval - either a normal evaluation or on-the-fly online training
        """
        # eval with training
        if self.self_supervised:
            return self.online_training(snr, gamma)
        else:
            # a normal evaluation, return evaluation of parent class
            return super().single_eval(snr, gamma)

    def online_training(self, snr: float, gamma: float) -> Union[float, np.ndarray]:
        self.check_eval_mode()
        self.load_weights(snr, gamma)
        self.deep_learning_setup()
        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)
        # self-supervised loop
        total_ser = 0
        ser_by_word = np.zeros(transmitted_words.shape[0])
        for count, (transmitted_word, received_word) in enumerate(zip(transmitted_words, received_words)):
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)

            # detect
            detected_word = self.detector(received_word, 'val')

            # decode
            decoded_word = [decode(detected_word, self.n_symbols) for detected_word in detected_word.cpu().numpy()]
            decoded_word = torch.Tensor(decoded_word).to(device)

            # calculate accuracy
            ser, fer, err_indices = calculate_error_rates(decoded_word, transmitted_word)

            # encode word again
            decoded_word_array = decoded_word.int().cpu().numpy()
            encoded_word = torch.Tensor(encode(decoded_word_array, self.n_symbols).reshape(1, -1)).to(device)
            errors_num = torch.sum(torch.abs(encoded_word - detected_word)).item()
            print('*' * 20)
            print(f'current: {count, ser, errors_num}')

            if ser <= SER_THRESH:
                # run training loops
                for i in range(SELF_SUPERVISED_ITERATIONS):
                    # calculate soft values
                    soft_estimation = self.detector(received_word, 'train')
                    labels = detected_word if ser > 0 else encoded_word
                    self.run_train_loop(soft_estimation=soft_estimation, transmitted_words=labels)

            total_ser += ser
            ser_by_word[count] = ser
            if (count + 1) % 10 == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')
        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        if self.eval_mode == 'by_word':
            return ser_by_word
        return total_ser

    def evaluate(self) -> np.ndarray:
        """
        Evaluation either happens in a point aggregation way, or in a word-by-word fashion
        """
        # eval with training
        if self.eval_mode == 'by_word' and self.self_supervised:
            snr = self.snr_range['val'][0]
            gamma = self.gamma_range[0]
            return self.online_training(snr, gamma)
        else:
            return super().evaluate()


if __name__ == '__main__':
    dec = METAVNETTrainer()
    dec.meta_train()
    # dec.evaluate()
    # dec.count_parameters()
