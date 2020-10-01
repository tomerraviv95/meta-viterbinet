import math

from python_code.channel.channel_dataset import ChannelModelDataset
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

MAX_RUNS = 10 ** 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # code parameters
        self.block_length = None
        self.code_length = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.use_ecc = None
        self.noisy_est_var = None

        # gamma
        self.gamma_start = None
        self.gamma_end = None
        self.gamma_num = None

        # validation hyperparameters
        self.val_channel_blocks = None
        self.val_SNR_start = None
        self.val_SNR_end = None
        self.val_SNR_step = None

        # training hyperparameters
        self.train_minibatch_num = None
        self.train_minibatch_size = None
        self.train_SNR_start = None
        self.train_SNR_end = None
        self.train_SNR_step = None
        self.train_channel_blocks = None
        self.lr = None  # learning rate
        self.loss_type = None
        self.print_every_n_train_minibatches = None
        self.optimizer_type = None
        self.early_stopping_mode = None

        # seed
        self.noise_seed = None
        self.word_seed = None

        # if any kwargs are passed, initialize the dict with them
        self.initialize_by_kwargs(**kwargs)

        # initializes all none parameters above from config
        self.param_parser(config_path)

        # initializes word and noise generator from seed
        self.rand_gen = np.random.RandomState(self.noise_seed)
        self.word_rand_gen = np.random.RandomState(self.word_seed)
        self.n_states = 2 ** self.memory_length
        self.transmission_length = self.block_length

        # initialize matrices, datasets and detector
        self.initialize_detector()
        self.initialize_dataloaders()

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
        if self.early_stopping_mode == 'on':
            self.es = EarlyStopping(patience=20)
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
        self.channel_blocks_per_phase = {'train': self.train_channel_blocks, 'val': self.val_channel_blocks}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       transmission_length=self.transmission_length,
                                       channel_blocks=self.channel_blocks_per_phase[phase],
                                       memory_length=self.memory_length,
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       noisy_est_var=self.noisy_est_var,
                                       use_ecc=self.use_ecc)
            for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val']}

    def load_weights(self, *args):
        """
        Loads detector's weights from highest checkpoint in run_name
        """
        pass

    def evaluate(self) -> np.ndarray:
        """
        Monte-Carlo simulation over validation SNRs range
        :return: ber, fer, iterations vectors
        """
        ser_total = np.zeros(len(self.snr_range['val']))
        with torch.no_grad():
            for snr_ind, snr in enumerate(self.snr_range['val']):
                print(f'Starts evaluation at snr {snr}')
                start = time()
                ser_snr = 0
                for gamma in self.gamma_range:
                    self.load_weights(snr, gamma)
                    ser_snr += self.single_eval(snr, gamma)

                # divide by the number of different gamma values
                ser_snr /= self.gamma_num

                ser_total[snr_ind] = ser_snr
                print(f'Done. time: {time() - start}, ser: {ser_snr}')

        return ser_total

    def single_eval(self, snr: float, gamma: float) -> float:
        """
        Evaluation at a single snr.
        :param snr: indice of snr in the snrs vector
        :return: ser for batch
        """

        # create state_estimator_morning data
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr=snr, gamma=gamma)
        transmitted_words = transmitted_words.to(device=device)
        received_words = received_words.to(device=device)

        # decode and calculate accuracy
        decoded_words = self.detector(received_words, 'val', gamma)
        ser, fer, err_indices = calculate_error_rates(decoded_words, transmitted_words)

        return ser

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
                    # run single train loop
                    loss = self.single_train_loop(snr, gamma)
                    current_loss = loss.item()
                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if minibatch % self.print_every_n_train_minibatches == 0:
                        # evaluate performance
                        ser = self.single_eval(snr, gamma)
                        print(f'Minibatch {minibatch}, Loss {current_loss}, ser - {ser}')
                        # save best weights
                        if ser < best_ser:
                            self.save_weights(current_loss, snr, gamma)
                            best_ser = ser
                        if self.es.step(ser):
                            break

                print(f'best ser - {best_ser}')
                print('*' * 50)

    def single_train_loop(self, snr: float, gamma: float) -> torch.Tensor:
        # draw words
        transmitted_words, received_words = self.channel_dataset['train'].__getitem__(snr=snr, gamma=gamma)
        transmitted_words = transmitted_words.to(device=device)
        received_words = received_words.to(device=device)

        # pass through detector
        soft_estimation = self.detector(received_words, 'train', gamma)

        # calculate loss
        loss = self.calc_loss(soft_estimation=soft_estimation, transmitted_words=transmitted_words)

        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('Nan value')
        return loss

    def save_weights(self, current_loss: float, snr: float, gamma: float):
        torch.save({'model_state_dict': self.detector.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss},
                   os.path.join(self.weights_dir, f'snr_{snr}_gamma_{gamma}.pt'))
