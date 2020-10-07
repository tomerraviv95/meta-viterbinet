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
                'ViterbiNet, perfect CSI (paper)': 's',
                'Viterbi, Initial CSI': 'x',
                'Viterbi, Full CSI': '^',
                'ViterbiNet, initial training': '>',
                'ViterbiNet, composite training': 'o',
                'ViterbiNet, online training': 's',
                'Viterbi, Initial CSI (paper)': 'x',
                'Viterbi, Full CSI (paper)': '^',
                'ViterbiNet, initial training (paper)': '>',
                'ViterbiNet, composite training (paper)': 'o',
                'ViterbiNet, online training (paper)': 's'
                }
COLORS_DICT = {'Viterbi, CSI uncertainty': 'black',
               'Viterbi, CSI uncertainty (paper)': 'black',
               'Viterbi, perfect CSI': 'blue',
               'Viterbi, perfect CSI (paper)': 'blue',
               'ViterbiNet, CSI uncertainty': 'green',
               'ViterbiNet, CSI uncertainty (paper)': 'green',
               'ViterbiNet, perfect CSI': 'red',
               'ViterbiNet, perfect CSI (paper)': 'red',
               'Viterbi, Initial CSI': 'black',
               'Viterbi, Full CSI': 'blue',
               'ViterbiNet, initial training': 'purple',
               'ViterbiNet, composite training': 'green',
               'ViterbiNet, online training': 'r',
               'Viterbi, Initial CSI (paper)': 'black',
               'Viterbi, Full CSI (paper)': 'blue',
               'ViterbiNet, initial training (paper)': 'purple',
               'ViterbiNet, composite training (paper)': 'green',
               'ViterbiNet, online training (paper)': 'r',
               }
LINESTYLES_DICT = {'Viterbi, CSI uncertainty': 'solid',
                   'Viterbi, perfect CSI': 'solid',
                   'ViterbiNet, CSI uncertainty': 'solid',
                   'ViterbiNet, perfect CSI': 'solid',
                   'Viterbi, Initial CSI': 'solid',
                   'Viterbi, Full CSI': 'solid',
                   'ViterbiNet, initial training': 'solid',
                   'ViterbiNet, composite training': 'solid',
                   'ViterbiNet, online training': 'solid',
                   'Viterbi, CSI uncertainty (paper)': 'dotted',
                   'Viterbi, perfect CSI (paper)': 'dotted',
                   'ViterbiNet, CSI uncertainty (paper)': 'dotted',
                   'ViterbiNet, perfect CSI (paper)': 'dotted',
                   'Viterbi, Initial CSI (paper)': 'dotted',
                   'Viterbi, Full CSI (paper)': 'dotted',
                   'ViterbiNet, initial training (paper)': 'dotted',
                   'ViterbiNet, composite training (paper)': 'dotted',
                   'ViterbiNet, online training (paper)': 'dotted',
                   }


def get_ser_plot(dec: Trainer, run_over: bool, method_name: str):
    print(method_name)
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
    dec = VATrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0.1,
                    gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN', channel_blocks=10)
    ser = get_ser_plot(dec, run_over=run_over, method_name=dec.get_name())
    all_curves.append((dec.snr_range['val'], ser, dec.get_name()))


def add_viterbi(all_curves):
    dec = VATrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0,
                    gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN')
    ser = get_ser_plot(dec, run_over=run_over, method_name=dec.get_name())
    all_curves.append((dec.snr_range['val'], ser, dec.get_name()))


def add_viterbi_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.058, 0.022, 5e-3, 5e-4]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'Viterbi, perfect CSI (paper)'))


def add_noisy_viterbinet(all_curves):
    dec = VNETTrainer(val_SNR_start=-6, val_SNR_end=10, noisy_est_var=0.1, gamma_start=0.1, gamma_end=2,
                      gamma_num=20, channel_type='ISI_AWGN',
                      weights_dir=os.path.join(WEIGHTS_DIR, 'paper_recreation_noisy'))
    ser = get_ser_plot(dec, run_over=run_over, method_name=dec.get_name())
    all_curves.append((dec.snr_range['val'], ser, dec.get_name()))


def add_viterbinet(all_curves):
    dec = VNETTrainer(val_SNR_start=-6, val_SNR_end=10, val_SNR_step=2, noisy_est_var=0,
                      gamma_start=0.1, gamma_end=2, gamma_num=20, channel_type='ISI_AWGN',
                      weights_dir=os.path.join(WEIGHTS_DIR, 'paper_recreation'))
    ser = get_ser_plot(dec, run_over=run_over, method_name=dec.get_name())
    all_curves.append((dec.snr_range['val'], ser, dec.get_name()))


def add_noisy_viterbinet_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.064, 0.027, 6.8e-3, 1.5e-3]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'ViterbiNet, CSI uncertainty (paper)'))


def add_viterbinet_paper(all_curves):
    snr_range = np.arange(-6, 11, 2)
    ser_awgn_perfect_from_paper = [0.31, 0.26, 0.21, 0.17, 0.11, 0.058, 0.023, 5.2e-3, 7e-4]
    all_curves.append((snr_range, ser_awgn_perfect_from_paper, 'ViterbiNet, perfect CSI (paper)'))


def add_viterbi_initial_csi(all_curves):
    dec = VATrainer(val_SNR_start=6, val_SNR_end=14, val_SNR_step=2, noisy_est_var=0,
                    fading_in_channel=True, fading_in_decoder=False, use_ecc=True, val_block_length=1784,
                    gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN')
    ser = get_ser_plot(dec, run_over=run_over, method_name='Viterbi, Initial CSI')
    all_curves.append((dec.snr_range['val'], ser, 'Viterbi, Initial CSI'))


def add_viterbi_full_csi(all_curves):
    dec = VATrainer(val_SNR_start=6, val_SNR_end=12, val_SNR_step=2, noisy_est_var=0,
                    fading_in_channel=True, fading_in_decoder=True, use_ecc=True, val_block_length=1784,
                    gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN')
    ser = get_ser_plot(dec, run_over=run_over, method_name='Viterbi, Full CSI')
    all_curves.append((dec.snr_range['val'], ser, 'Viterbi, Full CSI'))


def add_viterbinet_initial(all_curves):
    dec = VNETTrainer(val_SNR_start=6, val_SNR_end=14, val_SNR_step=2, val_block_length=1784,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=True, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      weights_dir=os.path.join(WEIGHTS_DIR, 'initial_viterbinet'), self_supervised=False)
    ser = get_ser_plot(dec, run_over=run_over, method_name='ViterbiNet, initial training')
    all_curves.append((dec.snr_range['val'], ser, 'ViterbiNet, initial training'))


def add_viterbinet_composite(all_curves):
    dec = VNETTrainer(val_SNR_start=6, val_SNR_end=14, val_SNR_step=2, val_block_length=1784,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      weights_dir=os.path.join(WEIGHTS_DIR, 'composite_viterbinet'), self_supervised=False)
    ser = get_ser_plot(dec, run_over=run_over, method_name='ViterbiNet, composite training')
    all_curves.append((dec.snr_range['val'], ser, 'ViterbiNet, composite training'))


def add_viterbinet_self_supervised(all_curves):
    dec = VNETTrainer(val_SNR_start=6, val_SNR_end=12, val_SNR_step=2, val_block_length=1784,
                      noisy_est_var=0, fading_in_channel=True, fading_in_decoder=False, use_ecc=True,
                      self_supervised=True, val_words=200,
                      gamma_start=0.2, gamma_end=0.2, gamma_num=1, channel_type='ISI_AWGN',
                      weights_dir=os.path.join(WEIGHTS_DIR, 'self_supervised_viterbinet'))
    ser = get_ser_plot(dec, run_over=run_over, method_name='ViterbiNet, online training')
    all_curves.append((dec.snr_range['val'], ser, 'ViterbiNet, online training'))


def add_viterbi_initial_csi_paper(all_curves):
    snr_range = np.arange(6, 14.1, 2)
    ser_paper = [0.11, 0.08, 0.048, 0.03, 0.02]
    all_curves.append((snr_range, ser_paper, 'Viterbi, Initial CSI (paper)'))


def add_viterbi_full_csi_paper(all_curves):
    snr_range = np.arange(6, 12.1, 2)
    ser_paper = [0.07, 0.026, 0.0065, 1.8e-4]
    all_curves.append((snr_range, ser_paper, 'Viterbi, Full CSI (paper)'))


def add_viterbinet_initial_paper(all_curves):
    snr_range = np.arange(6, 14.1, 2)
    ser_paper = [0.11, 0.085, 0.048, 0.033, 0.024]
    all_curves.append((snr_range, ser_paper, 'ViterbiNet, initial training (paper)'))


def add_viterbinet_composite_paper(all_curves):
    snr_range = np.arange(6, 14.1, 2)
    ser_paper = [0.08, 0.05, 0.028, 0.011, 0.005]
    all_curves.append((snr_range, ser_paper, 'ViterbiNet, composite training (paper)'))


def add_viterbinet_self_supervised_paper(all_curves):
    snr_range = np.arange(6, 12.1, 2)
    ser_paper = [0.11, 0.06, 0.038, 0.00045]
    all_curves.append((snr_range, ser_paper, 'ViterbiNet, online training (paper)'))


def get_figure_six_curves(all_curves):
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


if __name__ == '__main__':
    run_over = False
    all_curves = []

    # get_figure_six_curves(all_curves)

    add_viterbi_initial_csi(all_curves)
    add_viterbi_initial_csi_paper(all_curves)

    add_viterbi_full_csi(all_curves)
    add_viterbi_full_csi_paper(all_curves)

    add_viterbinet_initial(all_curves)
    add_viterbinet_initial_paper(all_curves)

    add_viterbinet_composite(all_curves)
    add_viterbinet_composite_paper(all_curves)

    add_viterbinet_self_supervised(all_curves)
    add_viterbinet_self_supervised_paper(all_curves)

    plot_all_curves(all_curves)
