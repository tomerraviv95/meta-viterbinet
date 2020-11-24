from typing import Tuple, Union

from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.ecc.rs_main import decode, encode
from python_code.utils.metrics import calculate_error_rates
from dir_definitions import CONFIG_PATH, WEIGHTS_DIR
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import RMSprop, Adam, SGD
from shutil import copyfile
import yaml
import torch
import os
from time import time
import numpy as np
import math
import copy

from python_code.utils.python_utils import copy_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ONLINE_META_FRAMES = 1


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # code parameters
        self.use_ecc = None
        self.n_symbols = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.channel_coefficients = None
        self.noisy_est_var = None
        self.fading_in_channel = None
        self.fading_in_decoder = None
        self.subframes_in_frame = None

        # gamma
        self.gamma_start = None
        self.gamma_end = None
        self.gamma_num = None

        # validation hyperparameters
        self.val_block_length = None
        self.val_frames = None
        self.val_SNR_start = None
        self.val_SNR_end = None
        self.val_SNR_step = None
        self.eval_mode = None

        # training hyperparameters
        self.train_block_length = None
        self.train_frames = None
        self.train_minibatch_num = None
        self.train_minibatch_size = None
        self.train_SNR_start = None
        self.train_SNR_end = None
        self.train_SNR_step = None
        self.lr = None  # learning rate
        self.loss_type = None
        self.print_every_n_train_minibatches = None
        self.optimizer_type = None

        # self-supervised online training
        self.self_supervised = None
        self.self_supervised_iterations = None
        self.ser_thresh = None
        self.online_meta = None

        # seed
        self.noise_seed = None
        self.word_seed = None

        # weights dir
        self.weights_dir = None

        # if any kwargs are passed, initialize the dict with them
        self.initialize_by_kwargs(**kwargs)

        # initializes all none parameters above from config
        self.param_parser(config_path)

        # initializes word and noise generator from seed
        self.rand_gen = np.random.RandomState(self.noise_seed)
        self.word_rand_gen = np.random.RandomState(self.word_seed)
        self.n_states = 2 ** self.memory_length

        # initialize matrices, datasets and detector
        self.initialize_dataloaders()
        self.initialize_detector()
        self.initialize_meta_detector()
        self.data_indices = torch.Tensor(list(filter(lambda x: x % self.subframes_in_frame != 0,
                                                     [i for i in
                                                      range(self.val_frames * self.subframes_in_frame)]))).long()

    def initialize_by_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def param_parser(self, config_path: str):
        """
        Parse the config, load all attributes into the trainer
        :param config_path: path to config
        """
        if config_path is None:
            config_path = CONFIG_PATH

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # set attribute of Trainer with every config item
        for k, v in self.config.items():
            try:
                if getattr(self, k) is None:
                    setattr(self, k, v)
            except AttributeError:
                pass

        if self.weights_dir is None:
            self.weights_dir = os.path.join(WEIGHTS_DIR, self.run_name)
            if not os.path.exists(self.weights_dir) and len(self.weights_dir):
                os.makedirs(self.weights_dir)
                # save config in output dir
                copyfile(config_path, os.path.join(self.weights_dir, "config.yaml"))

    def get_name(self):
        return self.__name__()

    def initialize_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.detector = None
        pass

    def initialize_meta_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.meta_detector = None
        pass

    def check_eval_mode(self):
        if self.eval_mode != 'aggregated' and self.eval_mode != 'by_word':
            raise ValueError("No such eval mode!!!")

    # calculate train loss
    def calc_loss(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=self.lr)
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=self.lr)
        elif self.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if self.loss_type == 'BCE':
            self.criterion = BCELoss().to(device)
        elif self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        elif self.loss_type == 'MSE':
            self.criterion = MSELoss().to(device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloaders(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.snr_range = {'train': np.arange(self.train_SNR_start, self.train_SNR_end + 1, step=self.train_SNR_step),
                          'val': np.arange(self.val_SNR_start, self.val_SNR_end + 1, step=self.val_SNR_step)}
        self.gamma_range = np.linspace(self.gamma_start, self.gamma_end, self.gamma_num)
        self.frames_per_phase = {'train': self.train_frames, 'val': self.val_frames}
        self.block_lengths = {'train': self.train_block_length, 'val': self.val_block_length}
        self.transmission_lengths = {'train': self.train_block_length,
                                     'val': self.val_block_length if not self.use_ecc else self.val_block_length + 8 * self.n_symbols}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       block_length=self.block_lengths[phase],
                                       transmission_length=self.transmission_lengths[phase],
                                       words=self.frames_per_phase[phase] * self.subframes_in_frame,
                                       memory_length=self.memory_length,
                                       channel_coefficients=self.channel_coefficients,
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       noisy_est_var=self.noisy_est_var,
                                       use_ecc=self.use_ecc,
                                       n_symbols=self.n_symbols,
                                       fading_in_channel=self.fading_in_channel,
                                       fading_in_decoder=self.fading_in_decoder,
                                       phase=phase)
            for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val']}

    def online_training(self, detected_word: torch.Tensor, encoded_word: torch.Tensor, received_word: torch.Tensor,
                        ser: float):
        pass

    def single_eval_at_point(self, snr: float, gamma: float) -> float:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """
        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)

        # decode and calculate accuracy
        detected_words = self.detector(received_words, 'val', snr, gamma)

        if self.use_ecc:
            decoded_words = [decode(detected_word, self.n_symbols) for detected_word in detected_words.cpu().numpy()]
            detected_words = torch.Tensor(decoded_words).to(device)

        ser, fer, err_indices = calculate_error_rates(detected_words[self.data_indices],
                                                      transmitted_words[self.data_indices])

        return ser

    def gamma_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single gamma value.
        :return: ser for batch.
        """
        ser_total = np.zeros(len(self.snr_range['val']))
        for snr_ind, snr in enumerate(self.snr_range['val']):
            self.load_weights(snr, gamma)
            ser_total[snr_ind] = self.single_eval_at_point(snr, gamma)
        return ser_total

    def evaluate_at_point(self) -> np.ndarray:
        """
        Monte-Carlo simulation over validation SNRs range
        :return: ber, fer, iterations vectors
        """
        ser_total = np.zeros(len(self.snr_range['val']))
        with torch.no_grad():
            for gamma_count, gamma in enumerate(self.gamma_range):
                print(f'Starts evaluation at gamma {gamma}')
                start = time()
                ser_total += self.gamma_eval(gamma)
                print(f'Done. time: {time() - start}, ser: {ser_total / (gamma_count + 1)}')
        ser_total /= self.gamma_num
        return ser_total

    def eval_by_word(self, snr: float, gamma: float) -> Union[float, np.ndarray]:
        if self.self_supervised:
            self.deep_learning_setup()
        total_ser = 0
        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)
        ser_by_word = np.zeros(transmitted_words.shape[0])
        # saved detector is used to initialize the decoder in meta learning loops
        self.saved_detector = copy.deepcopy(self.detector)
        # query for all detected words
        buffer_transmitted = torch.empty([0, received_words.shape[1]]).to(device)
        buffer_received = torch.empty([0, received_words.shape[1]]).to(device)
        support_idx = torch.empty([0]).long().to(device)
        query_idx = torch.empty([0]).long().to(device)
        for count, (transmitted_word, received_word) in enumerate(zip(transmitted_words, received_words)):
            transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)
            # detect
            detected_word = self.detector(received_word, 'val', snr, gamma)
            if count in self.data_indices:
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
                if self.self_supervised and ser <= self.ser_thresh:
                    self.online_training(detected_word, encoded_word, received_word, ser)
                total_ser += ser
                ser_by_word[count] = ser
            else:
                print('*' * 20)
                print(f'current: {count}, Pilot')
                # encode word again
                decoded_word_array = transmitted_word.int().cpu().numpy()
                encoded_word = torch.Tensor(encode(decoded_word_array, self.n_symbols).reshape(1, -1)).to(device)
                ser = 0
                if self.self_supervised and ser <= self.ser_thresh:
                    self.online_training(detected_word, encoded_word, received_word, ser)

            # save the encoded word in the buffer
            buffer_transmitted = torch.cat([buffer_transmitted, encoded_word])
            buffer_received = torch.cat([buffer_received, received_word])

            # save relevant support and query indices
            if self.online_meta and ser <= self.ser_thresh:
                # support_idx = torch.cat([support_idx, torch.LongTensor([count]).to(device)])
                # query_idx = torch.cat([query_idx, torch.LongTensor([count]).to(device)])
                if count > 0:
                    for i in range(1, self.subframes_in_frame + 1):
                        prev_indice = count - i
                        if ser_by_word[prev_indice] <= self.ser_thresh:
                            support_idx = torch.cat([support_idx, torch.LongTensor([prev_indice]).to(device)])
                            query_idx = torch.cat([query_idx, torch.LongTensor([count]).to(device)])

            if self.online_meta and count % (ONLINE_META_FRAMES * self.subframes_in_frame) == 0:
                print('meta-training')
                self.initialize_detector()
                # self.compare_parameters(self.saved_detector, self.detector)
                self.deep_learning_setup()
                last_x_mask = query_idx >= count  # + 1 - self.subframes_in_frame
                print(support_idx[last_x_mask], query_idx[last_x_mask])
                META_TRAINING_ITER = 250
                for i in range(META_TRAINING_ITER):
                    self.meta_train_loop(buffer_received, buffer_transmitted,
                                         support_idx[last_x_mask], query_idx[last_x_mask])
                copy_model(source_model=self.detector, dest_model=self.saved_detector)
                # adapt bias for current iteration
                if self.self_supervised:
                    self.online_training(detected_word, encoded_word, received_word, ser)

            if (count + 1) % 10 == 0:
                print(f'Self-supervised: {count + 1}/{transmitted_words.shape[0]}, SER {total_ser / (count + 1)}')

        total_ser /= transmitted_words.shape[0]
        print(f'Final ser: {total_ser}')
        return ser_by_word

    def evaluate(self) -> np.ndarray:
        """
        Evaluation either happens in a point aggregation way, or in a word-by-word fashion
        """
        # eval with training
        self.check_eval_mode()
        if self.eval_mode == 'by_word':
            if not self.use_ecc:
                raise ValueError('Only supports ecc')
            snr = self.snr_range['val'][0]
            gamma = self.gamma_range[0]
            self.load_weights(snr, gamma)
            return self.eval_by_word(snr, gamma)
        else:
            return self.evaluate_at_point()

    def meta_train(self):
        """
        Main meta-training loop. Runs in minibatches, each minibatch is split to pairs of following words.
        The pairs are comprised of (support,query) words.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        # initialize weights and loss
        for snr in self.snr_range['train']:
            for gamma in self.gamma_range:
                print(f'SNR - {snr}, Gamma - {gamma}')
                # initialize weights and loss
                self.initialize_detector()
                self.deep_learning_setup()
                best_ser = math.inf
                for minibatch in range(1, self.train_minibatch_num + 1):
                    # draw words from different channels
                    transmitted_words, received_words = self.channel_dataset['train'].__getitem__(
                        snr_list=[snr],
                        gamma=gamma)
                    support_idx = torch.arange(0, transmitted_words.shape[0] - 1).long()
                    query_idx = torch.arange(1, transmitted_words.shape[0]).long()
                    loss_query = self.meta_train_loop(received_words, transmitted_words, support_idx, query_idx)
                    if minibatch % self.print_every_n_train_minibatches == 0:
                        # evaluate performance
                        ser = self.single_eval_at_point(snr, gamma)
                        print(f'Minibatch {minibatch}, ser - {ser}')
                        # save best weights
                        if ser < best_ser:
                            self.save_weights(float(loss_query), snr, gamma)
                            best_ser = ser

    def meta_train_loop(self, received_words: torch.Tensor, transmitted_words: torch.Tensor, support_idx, query_idx):
        # divide the words to following pairs - (support,query)
        support_transmitted_words, support_received_words = transmitted_words[support_idx], received_words[
            support_idx]
        query_transmitted_words, query_received_words = transmitted_words[query_idx], received_words[query_idx]
        loss_supp, loss_query = math.inf, math.inf
        # meta-learning loop
        for word_ind in range(support_idx.shape[0]):
            # support words
            support_rx = support_received_words[word_ind].reshape(1, -1)
            support_tx = support_transmitted_words[word_ind].reshape(1, -1)

            # local update (with support set)
            para_list_detector = list(map(lambda p: p[0], zip(self.detector.parameters())))
            soft_estimation_supp = self.meta_detector(support_rx, 'train', para_list_detector)
            loss_supp = self.calc_loss(soft_estimation=soft_estimation_supp, transmitted_words=support_tx)

            # set create_graph to True for MAML, False for FO-MAML
            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=False)
            updated_para_list_detector = list(
                map(lambda p: p[1] - self.lr * p[0], zip(local_grad, para_list_detector)))

            # query words
            query_rx = query_received_words[word_ind].reshape(1, -1)
            query_tx = query_transmitted_words[word_ind].reshape(1, -1)

            # meta-update (with query set) should be same channel with support set
            soft_estimation_query = self.meta_detector(query_rx, 'train', updated_para_list_detector)
            loss_query = self.calc_loss(soft_estimation=soft_estimation_query, transmitted_words=query_tx)
            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False)

            ind_param = 0
            for param in self.detector.parameters():
                param.grad = None  # zero_grad
                param.grad = meta_grad[ind_param]
                ind_param += 1

            self.optimizer.step()

        return loss_query

    def train(self):
        """
        Main training loop. Runs in minibatches.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        # batches loop
        for snr in self.snr_range['train']:
            for gamma in self.gamma_range:
                print(f'SNR - {snr}, Gamma - {gamma}')

                # initialize weights and loss
                self.initialize_detector()
                self.deep_learning_setup()
                best_ser = math.inf

                for minibatch in range(1, self.train_minibatch_num + 1):
                    # draw words
                    transmitted_words, received_words = self.channel_dataset['train'].__getitem__(snr_list=[snr],
                                                                                                  gamma=gamma)
                    # run training loops
                    for i in range(self.train_frames * self.subframes_in_frame):
                        # pass through detector
                        soft_estimation = self.detector(received_words[i].reshape(1, -1), 'train')
                        current_loss = self.run_train_loop(soft_estimation, transmitted_words[i].reshape(1, -1))

                    if minibatch % self.print_every_n_train_minibatches == 0:
                        # evaluate performance
                        ser = self.single_eval_at_point(snr, gamma)
                        print(f'Minibatch {minibatch}, Loss {current_loss}, ser - {ser}')
                        # save best weights
                        if ser < best_ser:
                            self.save_weights(current_loss, snr, gamma)
                            best_ser = ser

                print(f'best ser - {best_ser}')
                print('*' * 50)

    def run_train_loop(self, soft_estimation: torch.Tensor, transmitted_words: torch.Tensor):
        # calculate loss
        loss = self.calc_loss(soft_estimation=soft_estimation, transmitted_words=transmitted_words)
        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('Nan value')
        current_loss = loss.item()
        # back propagation
        for param in self.detector.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()
        return current_loss

    def save_weights(self, current_loss: float, snr: float, gamma: float):
        torch.save({'model_state_dict': self.detector.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss},
                   os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'))

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

    def select_batch(self, gt_examples: torch.LongTensor, soft_estimation: torch.Tensor) -> Tuple[
        torch.LongTensor, torch.Tensor]:
        """
        Select a batch from the input and gt labels
        :param gt_examples: training labels
        :param soft_estimation: the soft approximation, distribution over states (per word)
        :return: selected batch from the entire "epoch", contains both labels and the NN soft approximation
        """
        rand_ind = torch.multinomial(torch.arange(gt_examples.shape[0]).float(),
                                     self.train_minibatch_size).long().to(device)
        return gt_examples[rand_ind], soft_estimation[rand_ind]

    def count_parameters(self):
        print(sum(p.numel() for p in self.detector.parameters() if p.requires_grad))

    def compare_parameters(self, detector1, detector2):
        cum_sum = 0
        for p1, p2 in zip(detector1.parameters(), detector2.parameters()):
            cum_sum += torch.sum(torch.abs(p1 - p2))
        print(cum_sum)
