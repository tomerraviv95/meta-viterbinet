from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.metrics import calculate_error_rates
import yaml
import torch
import os
from torch.optim import RMSprop
from torch.nn import CrossEntropyLoss, BCELoss
from time import time
from typing import Tuple
from dir_definitions import CONFIG_PATH, WEIGHTS_DIR
import numpy as np
from shutil import copyfile

MAX_RUNS = 10 ** 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # code parameters
        self.block_length = None
        self.channel_blocks = None
        self.code_length = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.use_ecc = None
        self.noisy_est_var = None

        # validation hyperparameters
        self.val_minibatch_size = None  # the more the merrier :)
        self.val_SNR_start = None
        self.val_SNR_end = None
        self.gamma_start = None
        self.gamma_end = None
        self.gamma_num = None
        self.thresh_errors = None  # monte-carlo error threshold per point

        # training hyperparameters
        self.num_of_minibatches = None
        self.train_minibatch_size = None
        self.train_SNR_start = None
        self.train_SNR_end = None
        self.lr = None  # learning rate
        self.load_from_checkpoint = None  # loads last checkpoint, if exists in the run_name folder
        self.loss_type = None

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
        self.transmission_length = self.block_length * self.channel_blocks

        # initialize matrices, datasets and decoder
        self.start_minibatch = 0
        self.load_detector()
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

    def load_detector(self):
        """
        Every trainer must have some base decoder model
        """
        self.decoder = None
        pass

    # calculate train loss
    def calc_loss(self, soft_estimation: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=self.lr)
        if self.loss_type == 'BCE':
            self.criterion = BCELoss().to(device)
        elif self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device=device)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def initialize_dataloaders(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.snr_range = {'train': np.arange(self.train_SNR_start, self.train_SNR_end + 1),
                          'val': np.arange(self.val_SNR_start, self.val_SNR_end + 1)}
        self.gamma_range = np.linspace(self.gamma_start, self.gamma_end, self.gamma_num)
        self.batches_size = {'train': self.train_minibatch_size, 'val': self.val_minibatch_size}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       transmission_length=self.transmission_length,
                                       batch_size=self.batches_size[phase],
                                       snr_range=self.snr_range[phase],
                                       gamma_range=self.gamma_range,
                                       memory_length=self.memory_length,
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       use_ecc=self.use_ecc)
            for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase])
                            for phase in ['train', 'val']}

    def load_last_checkpoint(self):
        """
        Loads decoder's weights from highest checkpoint in run_name
        """
        print(self.run_name)
        folder = os.path.join(os.path.join(WEIGHTS_DIR, self.run_name))
        names = []
        for file in os.listdir(folder):
            if file.startswith("checkpoint_"):
                names.append(int(file.split('.')[0].split('_')[1]))
        names.sort()
        if len(names) == 0:
            print("No checkpoints in run dir!!!")
            return

        self.start_minibatch = int(names[-1])
        if os.path.isfile(os.path.join(WEIGHTS_DIR, self.run_name, f'checkpoint_{self.start_minibatch}.pt')):
            print(f'loading model from minibatch {self.start_minibatch}')
            checkpoint = torch.load(os.path.join(WEIGHTS_DIR, self.run_name, f'checkpoint_{self.start_minibatch}.pt'))
            try:
                self.decoder.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'There is no checkpoint {self.start_minibatch} in run "{self.run_name}", starting from scratch')

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte-Carlo simulation over validation SNRs range
        :return: ber, fer, iterations vectors
        """
        ber_total, fer_total = np.zeros(len(self.snr_range['val'])), np.zeros(len(self.snr_range['val']))
        with torch.no_grad():
            for snr_ind, snr in enumerate(self.snr_range['val']):
                print(f'Starts evaluation at snr {snr}')
                start = time()

                ber_snr, fer_snr = self.single_snr_eval(snr_ind)
                ber_total[snr_ind] = ber_snr
                fer_total[snr_ind] = fer_snr
                print(f'Done. time: {time() - start}, ber: {ber_total[snr_ind]}, fer: {fer_total[snr_ind]}')

        return ber_total, fer_total

    def single_snr_eval(self, snr_ind: int) -> Tuple[float, float]:
        ber_snr, fer_snr = 0, 0
        for gamma_ind, gamma in enumerate(self.gamma_range):
            runs_num, err_count = 0, 0
            ber_gamma, fer_gamma = 0, 0
            # either stop when simulated enough errors, or reached a maximum number of runs
            while err_count < self.thresh_errors and runs_num < MAX_RUNS:
                ber, fer, current_err_count = self.single_eval(snr_ind, gamma_ind)
                ber_gamma += ber
                fer_gamma += fer
                err_count += current_err_count
                runs_num += 1.0

            # get ber and fer per gamma accurately by dividing the number of required runs
            ber_gamma /= runs_num
            fer_gamma /= runs_num

            # at last sum all gamma runs together
            ber_snr += ber_gamma
            fer_snr += fer_gamma

        # divide by the number of different gamma values
        ber_snr /= self.gamma_num
        fer_snr /= self.gamma_num
        return ber_snr, fer_snr

    def single_eval(self, snr_ind: int, gamma_ind: int) -> Tuple[float, float, int]:
        """
        Evaluation at a single snr.
        :param snr_ind: indice of snr in the snrs vector
        :return: ber and fer for batch, average iterations per word and number of errors in current batch
        """
        # create state_estimator_morning data
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_ind=snr_ind,
                                                                                    gamma_ind=gamma_ind)
        transmitted_words = transmitted_words.to(device=device)
        received_words = received_words.to(device=device)

        # decode and calculate accuracy
        decoded_words = self.decoder(received_words, 'val', self.snr_range['val'][snr_ind], self.gamma_range[gamma_ind])
        ber, fer, err_indices = calculate_error_rates(decoded_words, transmitted_words)
        current_err_count = err_indices.shape[0]

        return ber, fer, current_err_count

    def train(self):
        """
        Main training loop. Runs in minibatches.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        self.deep_learning_setup()
        self.evaluate()

        # batches loop
        for minibatch in range(self.start_minibatch, self.num_of_minibatches + 1):
            print(f"Minibatch number - {str(minibatch)}")

            current_loss = 0

            # run single train loop
            current_loss += self.run_single_train_loop()

            print(f"Loss {current_loss}")

            # save weights
            self.save_checkpoint(current_loss, minibatch)

            # evaluate performance
            self.evaluate()

    def run_single_train_loop(self) -> float:
        # draw words
        transmitted_words, received_words = iter(self.channel_dataset['train'][:len(self.snr_range['train'])])
        transmitted_words = transmitted_words.to(device=device)
        received_words = received_words.to(device=device)

        # pass through decoder
        soft_estimation = self.decoder(received_words, 'train')

        # calculate loss
        loss = self.calc_loss(soft_estimation=soft_estimation, labels=transmitted_words)
        loss_val = loss.item()

        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('Nan value')

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_val

    def save_checkpoint(self, current_loss: float, minibatch: int):
        torch.save({'minibatch': minibatch,
                    'model_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss,
                    'lr': self.lr},
                   os.path.join(self.weights_dir, 'checkpoint_' + str(minibatch) + '.pt'))
