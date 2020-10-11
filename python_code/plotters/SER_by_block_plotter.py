from dir_definitions import FIGURES_DIR, WEIGHTS_DIR
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime
import math
import os
import itertools
from python_code.plotters.SER_plotter import get_ser_plot
from python_code.trainers.RNN.rnn_trainer import RNNTrainer
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.VNET.vnet_trainer import VNETTrainer

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [8.2, 6.45]
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


def plot_all_curves(all_curves: List[Tuple[np.ndarray, np.ndarray, str]]):
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
        block_range = np.arange(len(ser))
        c = 'black' if np.sum(ser > 0) < 7 else 'red'
        plt.plot(block_range, ser, label=method_name, marker='o', color=c,
                 linestyle='solid', linewidth=2.2, markersize=12)
        min_block_ind = block_range[0] if block_range[0] < min_block_ind else min_block_ind
        max_block_ind = block_range[-1] if block_range[-1] > max_block_ind else max_block_ind

    plt.ylabel('SER')
    plt.xlabel('Block Index')
    plt.grid(which='both', ls='--')
    plt.xlim([min_block_ind - 0.1, max_block_ind + 0.1])
    plt.legend(loc='lower left', prop={'size': 15})
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER_by_block.png'), bbox_inches='tight')
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


def add_viterbi_failure(all_curves):
    val_block_lengths = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    n_symbols = [1, 2, 3, 4, 5, 6, 7, 8]
    for val_block_length in val_block_lengths:
        for n_symbol in n_symbols:
            print(val_block_length, n_symbol)
            dec = VNETTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                              noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                              gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                              self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                              weights_dir=os.path.join(WEIGHTS_DIR, 'self_supervised_viterbinet_failure'))
            method_name = f'ViterbiNet - Block Length {val_block_length}, Error symbols {n_symbol}'
            ser = get_ser_plot(dec, run_over=run_over, method_name=method_name)
            all_curves.append((ser, method_name, val_block_length, n_symbol))

def add_rnn_failure(all_curves):
    val_block_lengths = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    n_symbols = [1, 2, 3, 4, 5, 6, 7, 8]
    for val_block_length in val_block_lengths:
        for n_symbol in n_symbols:
            print(val_block_length, n_symbol)
            dec = RNNTrainer(val_SNR_start=12, val_SNR_end=12, val_SNR_step=2, val_block_length=val_block_length,
                             noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                             gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                             self_supervised=True, val_words=100, eval_mode='by_word', n_symbols=n_symbol,
                             weights_dir=os.path.join(WEIGHTS_DIR, 'rnn_gamma_0.2'))
            method_name = f'RNN - Block Length {val_block_length}, Error symbols {n_symbol}'
            ser = get_ser_plot(dec, run_over=run_over, method_name=method_name)
            all_curves.append((ser, method_name, val_block_length, n_symbol))


if __name__ == '__main__':
    run_over = False
    all_curves = []
    add_viterbi_failure(all_curves)
    add_rnn_failure(all_curves)
    # plot_all_curves(all_curves)
    plot_schematic(all_curves)
