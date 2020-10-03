import math

from python_code.trainers.VNET.vnet_trainer import VNETTrainer
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.trainer import Trainer
from dir_definitions import FIGURES_DIR, PLOTS_DIR, WEIGHTS_DIR
import datetime
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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

MARKERS_DICT = {'Viterbi, CSI uncertainty': 'x',
                'Viterbi, CSI uncertainty (paper)': 'x',
                'Viterbi, perfect CSI': '^',
                'Viterbi, perfect CSI (paper)': '^',
                'ViterbiNet, CSI uncertainty': 'o',
                'ViterbiNet, CSI uncertainty (paper)': 'o',
                'ViterbiNet, perfect CSI': 's',
                'ViterbiNet, perfect CSI (paper)': 's'}
COLORS_DICT = {'Viterbi, CSI uncertainty': 'black',
               'Viterbi, CSI uncertainty (paper)': 'black',
               'Viterbi, perfect CSI': 'blue',
               'Viterbi, perfect CSI (paper)': 'blue',
               'ViterbiNet, CSI uncertainty': 'green',
               'ViterbiNet, CSI uncertainty (paper)': 'green',
               'ViterbiNet, perfect CSI': 'red',
               'ViterbiNet, perfect CSI (paper)': 'red'
               }
LINESTYLES_DICT = {'Viterbi, CSI uncertainty': 'solid',
                   'Viterbi, perfect CSI': 'solid',
                   'ViterbiNet, CSI uncertainty': 'solid',
                   'ViterbiNet, perfect CSI': 'solid',
                   'Viterbi, CSI uncertainty (paper)': 'dotted',
                   'Viterbi, perfect CSI (paper)': 'dotted',
                   'ViterbiNet, CSI uncertainty (paper)': 'dotted',
                   'ViterbiNet, perfect CSI (paper)': 'dotted'}


def get_ser_plot(dec: Trainer, run_over: bool):
    method_name = dec.get_name()

    # set the path to saved or needed-loading pkl file
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(dec.channel_type)])
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')

    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        ser_total = load_pkl(plots_path)
    else:
        print("calculating fresh")
        ser_total = dec.evaluate()
        save_pkl(plots_path, ser_total)

    return ser_total


def plot_all_curves(all_curves: List[Tuple[np.ndarray, np.ndarray, str]]):
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    min_snr = math.inf
    max_snr = -math.inf
    for snr_range, ber, method_name in all_curves:
        plt.plot(snr_range, ber, label=method_name, marker=MARKERS_DICT[method_name], color=COLORS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2, markersize=12)
        min_snr = snr_range[0] if snr_range[0] < min_snr else min_snr
        max_snr = snr_range[-1] if snr_range[-1] > max_snr else max_snr

    plt.yscale('log')
    plt.ylabel('SER')
    plt.xlabel('$E_b/N_0$ [dB]')
    plt.grid(which='both', ls='--')
    plt.xlim([min_snr - 0.1, max_snr + 0.1])
    plt.legend(loc='lower left', prop={'size': 15})
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER.png'), bbox_inches='tight')

    plt.show()


def add_noisy_viterbi_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_isi_from_paper = [0.31, 0.28, 0.25, 0.22, 0.18, 0.15, 0.12, 1e-1, 8.5e-2]
    all_curves.append((snr_range, ser_awgn_isi_from_paper, 'Viterbi, CSI uncertainty (paper)'))


def add_noisy_viterbi(all_curves):
    dec2 = VATrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0.1,
                     gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN', channel_blocks=10)
    ser2 = get_ser_plot(dec2, run_over=run_over)
    all_curves.append((dec2.snr_range['val'], ser2, dec2.get_name()))


def add_viterbi(all_curves):
    dec1 = VATrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0,
                     gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN')
    ser1 = get_ser_plot(dec1, run_over=run_over)
    all_curves.append((dec1.snr_range['val'], ser1, dec1.get_name()))


def add_viterbi_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.058, 0.022, 5e-3, 5e-4]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'Viterbi, perfect CSI (paper)'))


def add_noisy_viterbinet(all_curves):
    dec3 = VNETTrainer(val_SNR_start=-6, val_SNR_end=10, noisy_est_var=0.1, gamma_start=0.1, gamma_end=2,
                       gamma_num=20, channel_type='ISI_AWGN',
                       weights_dir=os.path.join(WEIGHTS_DIR, 'best_paper_recreation_noisy'))
    ser3 = get_ser_plot(dec3, run_over=run_over)
    all_curves.append((dec3.snr_range['val'], ser3, dec3.get_name()))


def add_viterbinet(all_curves):
    dec4 = VNETTrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0,
                       gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN',
                       weights_dir=os.path.join(WEIGHTS_DIR, 'best_paper_recreation'))
    ser4 = get_ser_plot(dec4, run_over=run_over)
    all_curves.append((dec4.snr_range['val'], ser4, dec4.get_name()))


def add_noisy_viterbinet_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.064, 0.027, 6.8e-3, 1.5e-3]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'ViterbiNet, CSI uncertainty (paper)'))


def add_viterbinet_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.058, 0.023, 5.2e-3, 7e-4]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'ViterbiNet, perfect CSI (paper)'))


if __name__ == '__main__':
    run_over = False
    all_curves = []

    # Viterbi - noisy estimate of CSI
    add_noisy_viterbi(all_curves)

    # Viterbi - noisy estimate of CSI (paper)
    add_noisy_viterbi_paper(all_curves)

    # Viterbi - perfect CSI
    add_viterbi(all_curves)

    # Viterbi - perfect CSI (paper)
    add_viterbi_paper(all_curves)

    # ViterbiNet - noisy
    add_noisy_viterbinet(all_curves)

    # ViterbiNet - noisy (paper)
    add_noisy_viterbinet_paper(all_curves)

    # ViterbiNet - perfect CSI
    add_viterbinet(all_curves)

    # ViterbiNet - perfect CSI (paper)
    add_viterbinet_paper(all_curves)

    plot_all_curves(all_curves)
