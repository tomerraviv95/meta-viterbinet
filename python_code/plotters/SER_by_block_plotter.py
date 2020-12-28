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
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'ViterbiNet': 'green',
               'LSTM': 'green',
               'Joint': 'orange',
               'JointRNN': 'blue',
               'Viterbi': 'black',
               'MetaRNN': 'r',
               'OnlineMetaRNN': 'black',
               'OnlineMetaViterbiNet': 'blue'}

MARKERS_DICT = {'ViterbiNet': 'd',
                'LSTM': 'd',
                'Joint': 'x',
                'JointRNN': 'x',
                'Viterbi': 'o',
                'MetaRNN': '.',
                'OnlineMetaRNN': 'x',
                'OnlineMetaViterbiNet': '.'}

LINESTYLES_DICT = {'ViterbiNet': 'solid',
                   'LSTM': 'dotted',
                   'Joint': 'solid',
                   'JointRNN': 'dotted',
                   'Viterbi': 'solid',
                   'MetaRNN': 'dotted',
                   'OnlineMetaRNN': 'solid',
                   'OnlineMetaViterbiNet': 'dotted'}

METHOD_NAMES = {'ViterbiNet': 'ViterbiNet, online training',
                'LSTM': 'LSTM, online training',
                'Joint': 'ViterbiNet, joint training',
                'JointRNN': 'LSTM, joint training',
                'Viterbi': 'Viterbi, full CSI',
                'MetaRNN': 'Meta LSTM',
                'OnlineMetaRNN': 'Online Meta LSTM',
                'OnlineMetaViterbiNet': 'Online Meta ViterbiNet'}


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
        block_range = np.arange(1, len(ser) + 1)
        key = method_name.split(' ')[0]
        plt.plot(block_range, ser, label=METHOD_NAMES[key], color=COLORS_DICT[key], marker=MARKERS_DICT[key],
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


def plot_all_curves_aggregated(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], val_block_length, n_symbol, snr):
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
        block_range = np.arange(1, len(ser) + 1)
        key = method_name.split(' ')[0]
        plt.plot(block_range, np.cumsum(ser) / np.arange(1, len(ser) + 1), label=METHOD_NAMES[key],
                 color=COLORS_DICT[key], marker=MARKERS_DICT[key],
                 linestyle=LINESTYLES_DICT[key], linewidth=2.2)
        min_block_ind = block_range[0] if block_range[0] < min_block_ind else min_block_ind
        max_block_ind = block_range[-1] if block_range[-1] > max_block_ind else max_block_ind
    plt.ylabel('Coded BER')
    plt.xlabel('Block Index')
    plt.xlim([min_block_ind - 0.1, max_block_ind + 0.1])
    plt.legend(loc='upper left')
    plt.savefig(
        os.path.join(FIGURES_DIR, folder_name,
                     f'SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}.png'),
        bbox_inches='tight')
    # plt.show()


def plot_schematic(all_curves, val_block_lengths):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    names = list(set([all_curves[i][1] for i in range(len(all_curves))]))
    for method_name in names:
        mean_sers = []
        key = method_name.split(' ')[0]
        for ser, cur_name, val_block_length, n_symbol in all_curves:
            mean_ser = np.mean(ser)
            if cur_name != method_name:
                continue
            mean_sers.append(mean_ser)
        plt.plot(val_block_lengths, mean_sers, label=METHOD_NAMES[key],
                 color=COLORS_DICT[key], marker=MARKERS_DICT[key],
                 linestyle=LINESTYLES_DICT[key], linewidth=2.2)

    plt.xticks(val_block_lengths, val_block_lengths)
    plt.xlabel('SNR[dB]')
    plt.ylabel('Coded BER')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper right', prop={'size': 15})
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_block_length.png'),
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


def add_joint_viterbinet(all_curves, val_block_length, n_symbol, snr):
    dec = VNETTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=True,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='time_decay',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=2,
                      self_supervised=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'Joint ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_rnn(all_curves, val_block_length, n_symbol, snr):
    dec = LSTMTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=True,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='time_decay',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=2,
                      self_supervised=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'JointRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_viterbinet(all_curves, val_block_length, n_symbol, snr):
    dec = VNETTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=True,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='time_decay',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=2,
                      self_supervised=True,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_rnn(all_curves, val_block_length, n_symbol, snr):
    dec = LSTMTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=True,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='time_decay',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=2,
                      self_supervised=True,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'LSTM'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_onlinemetaviterbinet(all_curves, val_block_length, n_symbol, snr):
    dec = METAVNETTrainer(val_SNR_start=snr,
                          val_SNR_end=snr,
                          train_SNR_start=snr,
                          train_SNR_end=snr,
                          val_SNR_step=2,
                          train_SNR_step=2,
                          val_block_length=val_block_length,
                          train_block_length=val_block_length,
                          noisy_est_var=0,
                          fading_in_channel=True,
                          fading_in_decoder=False,
                          use_ecc=True,
                          gamma_start=0.2,
                          gamma_end=0.2,
                          gamma_num=1,
                          channel_type='ISI_AWGN',
                          channel_coefficients='time_decay',
                          subframes_in_frame=25,
                          val_frames=VAL_FRAMES,
                          eval_mode='by_word',
                          n_symbols=n_symbol,
                          fading_taps_type=2,
                          self_supervised_iterations=200,
                          self_supervised=True,
                          ser_thresh=0.02,
                          online_meta=True,
                          buffer_empty=True,
                          weights_init='last_frame',
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'meta_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'OnlineMetaViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_online_metarnn(all_curves, val_block_length, n_symbol, snr):
    dec = MetaLSTMTrainer(val_SNR_start=snr,
                          val_SNR_end=snr,
                          train_SNR_start=snr,
                          train_SNR_end=snr,
                          val_SNR_step=2,
                          train_SNR_step=2,
                          val_block_length=val_block_length,
                          train_block_length=val_block_length,
                          noisy_est_var=0,
                          fading_in_channel=True,
                          fading_in_decoder=False,
                          use_ecc=True,
                          gamma_start=0.2,
                          gamma_end=0.2,
                          gamma_num=1,
                          channel_type='ISI_AWGN',
                          channel_coefficients='time_decay',
                          subframes_in_frame=25,
                          val_frames=VAL_FRAMES,
                          eval_mode='by_word',
                          n_symbols=n_symbol,
                          fading_taps_type=2,
                          self_supervised_iterations=200,
                          self_supervised=True,
                          ser_thresh=0.02,
                          online_meta=True,
                          buffer_empty=True,
                          weights_init='last_frame',
                          meta_train_iterations=10,
                          meta_j_num=5,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'rnn_meta_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'OnlineRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


VAL_FRAMES = 8

if __name__ == '__main__':
    run_over = True
    # val_block_lengths = [80, 120, 160, 200, 240, 280]
    val_block_lengths = [200]
    # snr_values = [12]
    snr_values = [6, 7, 8, 9, 10, 11, 12, 13, 14]
    n_symbols = [2]
    all_curves = []
    for snr in snr_values:
        for val_block_length in val_block_lengths:
            for n_symbol in n_symbols:
                print(val_block_length, n_symbol)
                # add_joint_viterbinet(all_curves, val_block_length, n_symbol, snr)
                # add_viterbinet(all_curves, val_block_length, n_symbol, snr)
                # add_onlinemetaviterbinet(all_curves, val_block_length, n_symbol, snr)
                #
                # add_joint_rnn(all_curves, val_block_length, n_symbol, snr)
                # add_rnn(all_curves, val_block_length, n_symbol, snr)
                add_online_metarnn(all_curves, val_block_length, n_symbol, snr)

                # plot_all_curves_aggregated(all_curves, val_block_length, n_symbol, snr)
                # plot_all_curves(all_curves, val_block_length, n_symbol)
    plot_schematic(all_curves, snr_values)
