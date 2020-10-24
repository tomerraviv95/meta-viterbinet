from typing import Tuple

from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.ecc.rs_main import decode
from python_code.utils.early_stopping import EarlyStopping
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS_NUM = 10


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # code parameters
        self.channel_blocks = None
        self.use_ecc = None
        self.n_symbols = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.noisy_est_var = None
        self.fading_in_channel = None
        self.fading_in_decoder = None

        # gamma
        self.gamma_start = None
        self.gamma_end = None
        self.gamma_num = None

        # validation hyperparameters
        self.val_block_length = None
        self.val_SNR_start = None
        self.val_SNR_end = None
        self.val_SNR_step = None
        self.val_words = None
        self.eval_mode = None

        # training hyperparameters
        self.train_block_length = None
        self.train_minibatch_num = None
        self.train_minibatch_size = None
        self.train_SNR_start = None
        self.train_SNR_end = None
        self.train_SNR_step = None
        self.lr = None  # learning rate
        self.loss_type = None
        self.print_every_n_train_minibatches = None
        self.optimizer_type = None
        self.early_stopping_mode = None
        self.self_supervised = None

        # meta
        self.meta_words = None
        self.meta_lr = None
        self.support_size = None

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
        if self.early_stopping_mode:
            self.es = EarlyStopping(patience=4)
        else:
            self.es = None
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
        self.channel_blocks_per_phase = {'train': self.channel_blocks,
                                         'val': self.channel_blocks,
                                         'meta_train': self.channel_blocks}
        self.words_per_phase = {'train': 1, 'val': self.val_words, 'meta_train': self.meta_words}
        self.block_lengths = {'train': self.train_block_length,
                              'val': self.val_block_length,
                              'meta_train': self.train_block_length}
        self.transmission_lengths = {
            'train': self.train_block_length,
            'val': self.val_block_length if not self.use_ecc else self.val_block_length + 8 * self.n_symbols,
            'meta_train': self.train_block_length}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       block_length=self.block_lengths[phase],
                                       transmission_length=self.transmission_lengths[phase],
                                       channel_blocks=self.channel_blocks_per_phase[phase],
                                       words=self.words_per_phase[phase],
                                       memory_length=self.memory_length,
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       noisy_est_var=self.noisy_est_var,
                                       use_ecc=self.use_ecc,
                                       n_symbols=self.n_symbols,
                                       fading_in_channel=self.fading_in_channel,
                                       fading_in_decoder=self.fading_in_decoder,
                                       phase=phase)
            for phase in ['train', 'val', 'meta_train']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val', 'meta_train']}

    def single_eval(self, snr: float, gamma: float) -> float:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """
        # draw words of given gamma for all snrs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[snr], gamma=gamma)

        # decode and calculate accuracy
        detected_words = self.detector(received_words, 'val')

        if self.use_ecc:
            decoded_words = [decode(detected_word, self.n_symbols) for detected_word in detected_words.cpu().numpy()]
            detected_words = torch.Tensor(decoded_words).to(device)

        ser, fer, err_indices = calculate_error_rates(detected_words, transmitted_words)

        return ser

    def gamma_eval(self, gamma: float) -> np.ndarray:
        """
        Evaluation at a single gamma value.
        :return: ser for batch.
        """
        ser_total = np.zeros(len(self.snr_range['val']))

        for snr_ind, snr in enumerate(self.snr_range['val']):
            self.load_weights(snr, gamma)
            ser_total[snr_ind] = self.single_eval(snr, gamma)

        return ser_total

    def evaluate(self) -> np.ndarray:
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

    def meta_train(self):
        # initialize weights and loss
        for snr in self.snr_range['train']:
            for gamma in self.gamma_range:
                print(f'SNR - {snr}, Gamma - {gamma}')
                # initialize weights and loss
                self.initialize_detector()
                self.deep_learning_setup()
                best_ser = math.inf
                for minibatch in range(1, self.train_minibatch_num + 1):
                    # draw words
                    # at each minibatch, use different channel
                    support_transmitted_words, support_received_words = self.channel_dataset['meta_train'].__getitem__(
                        snr_list=[snr],
                        gamma=gamma)
                    query_transmitted_words, query_received_words = self.channel_dataset['meta_train'].__getitem__(
                        snr_list=[snr],
                        gamma=gamma)
                    loss_supp, loss_query = math.inf, math.inf
                    for word_ind in range(self.meta_words):
                        if word_ind % 5 == 0:
                            support_rx = support_received_words[word_ind].reshape(1, -1)
                            support_tx = support_transmitted_words[word_ind].reshape(1, -1)

                            # local update (with support set)
                            para_list_detector = list(map(lambda p: p[0], zip(self.detector.parameters())))
                            soft_estimation_supp = self.meta_detector(support_rx, 'train',
                                                                      para_list_detector)
                            loss_supp = self.calc_loss(soft_estimation=soft_estimation_supp,
                                                       transmitted_words=support_tx)
                            #local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=False)
                            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=True) # if create_graph=False, it will become first-order MAML which is an approximated version of MAML (here MAML is the scheme that I chose for meta-learning :))

                            updated_para_list_detector = list(
                                map(lambda p: p[1] - self.meta_lr * p[0], zip(local_grad, para_list_detector)))

                            query_rx = query_received_words[word_ind].reshape(1, -1)
                            query_tx = query_transmitted_words[word_ind].reshape(1, -1)
                            # meta-update (with query set) should be same channel with support set
                            soft_estimation_query = self.meta_detector(query_rx, 'train',
                                                                       updated_para_list_detector)
                            loss_query = self.calc_loss(soft_estimation=soft_estimation_query,
                                                        transmitted_words=query_tx)
                            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False) # here it is fine to have create_graph=False since we are not taking any further derivatives w.r.t meta_grad

                            ind_param = 0
                            for param in self.detector.parameters():
                                param.grad = None  # zero_grad
                                param.grad = meta_grad[ind_param]
                                ind_param += 1

                            self.optimizer.step()

                    # evaluate performance
                    ser = self.single_eval(snr, gamma)
                    print(f'Minibatch {minibatch}, ser - {ser}')
                    # save best weights
                    if ser < best_ser:
                        self.save_weights(float(loss_query), snr, gamma)
                        best_ser = ser
                    if self.early_stopping_mode and self.es.step(ser):
                        break

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
                    for i in range(STEPS_NUM):
                        # pass through detector
                        soft_estimation = self.detector(received_words, 'train')
                        current_loss = self.run_train_loop(soft_estimation, transmitted_words)

                    if minibatch % self.print_every_n_train_minibatches == 0:
                        # evaluate performance
                        ser = self.single_eval(snr, gamma)
                        print(f'Minibatch {minibatch}, Loss {current_loss}, ser - {ser}')
                        # save best weights
                        if ser < best_ser:
                            self.save_weights(current_loss, snr, gamma)
                            best_ser = ser
                        if self.early_stopping_mode and self.es.step(ser):
                            break

                print(f'best ser - {best_ser}')
                print('*' * 50)

    def run_train_loop(self, soft_estimation, transmitted_words):
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
