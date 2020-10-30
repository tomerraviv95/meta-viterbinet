from dir_definitions import FIGURES_DIR, WEIGHTS_DIR
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime
import math
import os
from python_code.plotters.SER_plotter import get_ser_plot
from python_code.trainers.META_LSTM.meta_lstm_trainer import MetaLSTMTrainer
from python_code.trainers.META_VNET.metavnet_trainer import METAVNETTrainer
from python_code.trainers.LSTM.lstm_trainer import LSTMTrainer
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.VNET.vnet_trainer import VNETTrainer

mpl.rcParams['xtick.labelsize'] = 34
mpl.rcParams['ytick.labelsize'] = 34
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [8.2, 6.45]
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['axes.labelsize'] = 40
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['legend.fontsize'] = 26
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'ViterbiNet': 'green',
               'LSTM': 'green',
               'MetaViterbiNet': 'r',
               'Joint': 'blue',
               'JointRNN': 'blue',
               'Viterbi': 'black',
               'MetaRNN': 'r'}

MARKERS_DICT = {'ViterbiNet': 'd',
               'LSTM': 'd',
               'MetaViterbiNet': '.',
               'Joint': 'x',
               'JointRNN': 'x',
               'Viterbi': 'o',
               'MetaRNN': '.'}

LINESTYLES_DICT = {'ViterbiNet': 'solid',
               'LSTM': 'dotted',
               'MetaViterbiNet': 'solid',
               'Joint': 'solid',
               'JointRNN': 'dotted',
               'Viterbi': 'solid',
               'MetaRNN': 'dotted'}

METHOD_NAMES = {'ViterbiNet': 'ViterbiNet, online training',
               'LSTM': 'LSTM, online training',
               'MetaViterbiNet': 'Meta-ViterbiNet',
               'Joint': 'ViterbiNet, joint training',
               'JointRNN': 'LSTM, joint training',
               'Viterbi': 'Viterbi, full CSI',
               'MetaRNN': 'Meta-LSTM'}

def plot_all_curves(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], val_block_length, n_symbol):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    min_block_ind = math.inf
    max_block_ind = -math.inf
    # iterate all curves, plot each one
    for ser, method_name, _, _ in all_curves:
        print(method_name)
        block_range = np.arange(1,len(ser)+1)
        key = method_name.split(' ')[0]
        plt.plot(block_range, ser, label=METHOD_NAMES[key], color=COLORS_DICT[key],marker=MARKERS_DICT[key],
                 linestyle=LINESTYLES_DICT[key], linewidth=2.2)
        min_block_ind = block_range[0] if block_range[0] < min_block_ind else min_block_ind
        max_block_ind = block_range[-1] if block_range[-1] > max_block_ind else max_block_ind

    plt.ylabel('Coded BER')
    plt.xlabel('Block Index')
    # plt.grid(which='both', ls='-')
    plt.xlim([min_block_ind - 0.1, max_block_ind + 0.1])
    # plt.ylim([0,0.08])
    plt.legend(loc='upper left')
    plt.savefig(
        os.path.join(FIGURES_DIR, folder_name, f'Block Length {val_block_length}, Error symbols {n_symbol}.png'),
        bbox_inches='tight')
    plt.show()


def plot_schematic(all_curves):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    ser_thresh = 0
    plt.figure()
    val_block_lengths = []
    n_symbols = []
    colors = []
    for ser, _, val_block_length, n_symbol in all_curves:
        c = 'black' if np.sum(ser > 0) < 7 else 'red'
        val_block_lengths.append(val_block_length)
        n_symbols.append(n_symbol)
        colors.append(c)
    plt.scatter(val_block_lengths, n_symbols, c=colors)
    plt.xticks(val_block_lengths, val_block_lengths)
    plt.xlabel('Block Length')
    plt.ylabel('Num of Symbols')
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'block_length_versus_symbols_num.png'),
                bbox_inches='tight')
    plt.show()


def add_viterbi(all_curves, val_block_length, n_symbol):
    dec = VATrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                    noisy_est_var=0, fading_in_channel=True, fading_in_decoder=True, use_ecc=True,
                    gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                    self_supervised=False, val_words=100, eval_mode='by_word', n_symbols=n_symbol)
    method_name = f'Viterbi - Full CSI'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_viterbinet(all_curves, val_block_length, n_symbol):
    dec = VNETTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                      weights_dir=os.path.join(WEIGHTS_DIR, 'self_supervised_viterbinet_failure'))
    method_name = f'ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_rnn_online(all_curves, val_block_length, n_symbol):
    dec = LSTMTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                      weights_dir=os.path.join(WEIGHTS_DIR, 'rnn_gamma_0.2'))
    method_name = f'LSTM'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_metaviterbinet(all_curves, val_block_length, n_symbol):
    dec = METAVNETTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                          noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                          gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                          self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                          weights_dir=os.path.join(WEIGHTS_DIR, f'meta_training_{val_block_length}'))
    method_name = f'MetaViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_metarnn(all_curves, val_block_length, n_symbol):
    dec = MetaLSTMTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                          noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                          gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                          self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                          weights_dir=os.path.join(WEIGHTS_DIR, f'rnn_meta_training_{val_block_length}'))
    method_name = f'MetaRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_viterbinet(all_curves, val_block_length, n_symbol):
    dec = VNETTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      self_supervised=False, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                      weights_dir=os.path.join(WEIGHTS_DIR, f'viterbinet_joint_{val_block_length}'))
    method_name = f'Joint ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_rnn(all_curves, val_block_length, n_symbol):
    dec = LSTMTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      self_supervised=False, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                      weights_dir=os.path.join(WEIGHTS_DIR, f'rnn_joint_{val_block_length}'))
    method_name = f'JointRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_metaviterbinet(all_curves, val_block_length, n_symbol):
    dec = METAVNETTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                          noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                          gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                          self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                          weights_dir=os.path.join(WEIGHTS_DIR, f'viterbinet_joint_{val_block_length}'))
    method_name = f'JointViterbiNetMetaStrategy'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_metarnn(all_curves, val_block_length, n_symbol):
    dec = MetaLSTMTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                          noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                          gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                          self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                          weights_dir=os.path.join(WEIGHTS_DIR, f'rnn_joint_{val_block_length}'))
    method_name = f'JointRNNMetaStrategy'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


if __name__ == '__main__':
    run_over = False
    # val_block_lengths = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    # n_symbols = [1, 2, 3, 4, 5, 6, 7, 8]
    parameters = [(160, 5), (200, 7)]
    for val_block_length, n_symbol in parameters:
        all_curves = []
        print(val_block_length, n_symbol)
        add_rnn_online(all_curves, val_block_length, n_symbol)
        add_viterbinet(all_curves, val_block_length, n_symbol)
        add_joint_rnn(all_curves, val_block_length, n_symbol)
        add_joint_viterbinet(all_curves, val_block_length, n_symbol)
        add_metarnn(all_curves, val_block_length, n_symbol)
        add_metaviterbinet(all_curves, val_block_length, n_symbol)
        add_viterbi(all_curves, val_block_length, n_symbol)
        plot_all_curves(all_curves, val_block_length, n_symbol)
        # add_joint_metaviterbinet(all_curves, val_block_length, n_symbol)
        # add_joint_metarnn(all_curves, val_block_length, n_symbol)
