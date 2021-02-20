from dir_definitions import WEIGHTS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_schematic
from python_code.trainers.META_LSTM.meta_lstm_trainer import MetaLSTMTrainer
from python_code.trainers.META_VNET.metavnet_trainer import METAVNETTrainer
from python_code.trainers.LSTM.lstm_trainer import LSTMTrainer
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.VNET.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *
import os


def add_viterbi(all_curves, current_params):
    dec = VATrainer(self_supervised=False,
                    online_meta=False,
                    weights_dir=os.path.join(WEIGHTS_DIR,
                                             f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                    **HYPERPARAMS_DICT)
    method_name = f'Viterbi - Full CSI'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_joint_viterbinet(all_curves, current_params):
    dec = VNETTrainer(self_supervised=False,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                      **HYPERPARAMS_DICT)
    method_name = f'Joint ViterbiNet'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_joint_rnn(all_curves, current_params):
    dec = LSTMTrainer(self_supervised=False,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                      **HYPERPARAMS_DICT)
    method_name = f'JointRNN'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_viterbinet(all_curves, current_params):
    dec = VNETTrainer(self_supervised=True,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                      **HYPERPARAMS_DICT)
    method_name = f'ViterbiNet'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_rnn(all_curves, current_params):
    dec = LSTMTrainer(self_supervised=True,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                      **HYPERPARAMS_DICT)
    method_name = f'LSTM'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_onlinemetaviterbinet(all_curves, current_params):
    dec = METAVNETTrainer(self_supervised=True,
                          online_meta=True,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'meta_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          **HYPERPARAMS_DICT)
    method_name = f'OnlineMetaViterbiNet'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_online_metarnn(all_curves, current_params):
    dec = MetaLSTMTrainer(self_supervised=True,
                          online_meta=True,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'rnn_meta_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          **HYPERPARAMS_DICT)
    method_name = f'OnlineRNN'
    print(method_name)
    ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


HYPERPARAMS_DICT = {'val_SNR_step': 2,
                    'train_SNR_step': 2,
                    'noisy_est_var': 0,
                    'fading_taps_type': 2,
                    'fading_in_decoder': True,
                    'use_ecc': True,
                    'gamma': 0.2,
                    'channel_type': 'ISI_AWGN',
                    'val_frames': 12,
                    'subframes_in_frame': 25,
                    'eval_mode': 'by_word',
                    'self_supervised_iterations': 200,
                    'ser_thresh': 0.02,
                    'buffer_empty': True,
                    'weights_init': 'last_frame',
                    }

if __name__ == '__main__':
    run_over = False
    plot_by_block = False  # either plot by block, or by SNR

    parameters = [(7, 120),
                  (8, 120),
                  (9, 120),
                  (10, 120),
                  (11, 120),
                  (12, 120)]
    n_symbol = 2
    channel_coefficients = 'cost2100'  # 'time_decay','cost2100'
    all_curves = []

    for snr, val_block_length in parameters:
        print(snr, val_block_length, n_symbol)

        HYPERPARAMS_DICT['n_symbols'] = n_symbol
        HYPERPARAMS_DICT['val_SNR_start'] = snr
        HYPERPARAMS_DICT['val_SNR_end'] = snr
        HYPERPARAMS_DICT['train_SNR_start'] = snr
        HYPERPARAMS_DICT['train_SNR_end'] = snr
        HYPERPARAMS_DICT['val_block_length'] = val_block_length
        HYPERPARAMS_DICT['train_block_length'] = val_block_length
        HYPERPARAMS_DICT['fading_in_channel'] = True if channel_coefficients == 'time_decay' else False
        HYPERPARAMS_DICT['channel_coefficients'] = channel_coefficients

        current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + str(HYPERPARAMS_DICT['val_SNR_start']) + '_' + \
                         str(HYPERPARAMS_DICT['val_block_length']) + '_' + str(HYPERPARAMS_DICT['n_symbols'])

        add_joint_viterbinet(all_curves, current_params)
        add_joint_rnn(all_curves, current_params)
        add_viterbinet(all_curves, current_params)
        add_rnn(all_curves, current_params)
        add_onlinemetaviterbinet(all_curves, current_params)
        add_online_metarnn(all_curves, current_params)
        add_viterbi(all_curves, current_params)

        if plot_by_block:
            plot_all_curves_aggregated(all_curves, val_block_length, n_symbol, snr)

    snr_values = [x[0] for x in parameters]
    if not plot_by_block:
        plot_schematic(all_curves, snr_values)
