import os

import numpy as np

from dir_definitions import WEIGHTS_DIR
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated, plot_schematic
from python_code.trainers.LSTM.lstm_trainer import LSTMTrainer
from python_code.trainers.META_LSTM.meta_lstm_trainer import MetaLSTMTrainer
from python_code.trainers.META_VNET.metavnet_trainer import METAVNETTrainer
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.VNET.vnet_trainer import VNETTrainer


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


def add_joint_viterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'Joint ViterbiNet'
    for trial in range(trial_num):
        dec = VNETTrainer(self_supervised=False,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          seed=trial + 1,

                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_mismatched_joint_viterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'Joint ViterbiNet - Mismatched SNRs'
    for trial in range(trial_num):
        dec = VNETTrainer(self_supervised=False,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_mismatched'),
                          seed=trial + 1,
                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_joint_rnn(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'JointRNN'
    for trial in range(trial_num):
        dec = LSTMTrainer(self_supervised=False,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'rnn_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          seed=trial + 1,
                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_viterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'ViterbiNet'
    for trial in range(trial_num):
        dec = VNETTrainer(self_supervised=True,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          seed=trial + 1,
                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_mismatched_viterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'ViterbiNet - Mismatched'
    for trial in range(trial_num):
        dec = VNETTrainer(self_supervised=True,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_mismatched'),
                          seed=trial + 1,
                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_rnn(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'LSTM'
    for trial in range(trial_num):
        dec = LSTMTrainer(self_supervised=True,
                          online_meta=False,
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'rnn_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                          seed=trial + 1,
                          **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_onlinemetaviterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'OnlineMetaViterbiNet'
    for trial in range(trial_num):
        dec = METAVNETTrainer(self_supervised=True,
                              online_meta=True,
                              weights_dir=os.path.join(WEIGHTS_DIR,
                                                       f'meta_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                              seed=trial + 1,
                              **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_mismatched_onlinemetaviterbinet(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'OnlineMetaViterbiNet - Mismatched'
    for trial in range(trial_num):
        dec = METAVNETTrainer(self_supervised=True,
                              online_meta=True,
                              weights_dir=os.path.join(WEIGHTS_DIR,
                                                       f'meta_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_mismatched'),
                              seed=trial + 1,
                              **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


def add_online_metarnn(all_curves, current_params, trial_num=5):
    total_ser = []
    method_name = f'OnlineRNN'
    for trial in range(trial_num):
        dec = MetaLSTMTrainer(self_supervised=True,
                              online_meta=True,
                              weights_dir=os.path.join(WEIGHTS_DIR,
                                                       f'rnn_meta_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1'),
                              seed=trial + 1,
                              **HYPERPARAMS_DICT)
        print(method_name)
        ser = get_ser_plot(dec, run_over=run_over, method_name=method_name + '_' + current_params + '_' + str(trial))
        total_ser.append(ser)
    avg_ser = np.average(total_ser)
    all_curves.append((avg_ser, method_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


HYPERPARAMS_DICT = {'val_SNR_step': 2,
                    'train_SNR_step': 2,
                    'noisy_est_var': 0,
                    'fading_taps_type': 1,
                    'fading_in_channel': True,
                    'fading_in_decoder': True,
                    'use_ecc': True,
                    'gamma': 0.2,
                    'val_frames': 12,
                    'subframes_in_frame': 25,
                    'eval_mode': 'by_word',
                    'self_supervised_iterations': 200,
                    'ser_thresh': 0.02,
                    'buffer_empty': True,
                    'weights_init': 'last_frame',
                    'meta_lr': 0.1,
                    'window_size': 1,
                    'meta_train_iterations': 20,
                    'meta_j_num': 10,
                    'meta_subframes': 5
                    }

if __name__ == '__main__':
    run_over = False
    plot_by_block = False  # either plot by block, or by SNR
    plot_type = 'plot_by_meta_frames'

    if plot_type == 'plot_by_SNR':
        parameters = [7, 8, 9, 10, 11, 12]
        xlabel = 'SNR [dB]'
    elif plot_type == 'plot_by_meta_frames':
        parameters = [5, 10, 15, 20]
        snr = 10
        xlabel = 'Meta-Learning Frequency F'
    elif plot_type == 'plot_by_self_supervised_iterations':
        parameters = [7, 8, 9, 10, 11, 12]
        xlabel = 'SNR [dB]'
    elif plot_type == 'plot_non_periodic':
        parameters = [8, 9, 10, 11, 12, 13]
        xlabel = 'SNR [dB]'
        val_frames = 8
        noise_seed = 10
        word_seed = 10
    else:
        raise ValueError("No such plot type!!!")
    n_symbol = 2
    val_block_length = 120
    channel_coefficients = 'time_decay'  # 'time_decay','cost2100','non_periodic'
    channel_type = 'ISI_AWGN'  # ISI_AWGN, NON_LINEAR_ISI_AWGN
    all_curves = []

    for params in parameters:
        print(params)

        if plot_type == 'plot_by_SNR':
            HYPERPARAMS_DICT['val_SNR_start'] = params
            HYPERPARAMS_DICT['val_SNR_end'] = params
            HYPERPARAMS_DICT['train_SNR_start'] = params
            HYPERPARAMS_DICT['train_SNR_end'] = params
        elif plot_type == 'plot_by_meta_frames':
            HYPERPARAMS_DICT['val_SNR_start'] = snr
            HYPERPARAMS_DICT['val_SNR_end'] = snr
            HYPERPARAMS_DICT['train_SNR_start'] = snr
            HYPERPARAMS_DICT['train_SNR_end'] = snr
            HYPERPARAMS_DICT['meta_subframes'] = params
        elif plot_type == 'plot_by_self_supervised_iterations':
            HYPERPARAMS_DICT['val_SNR_start'] = params
            HYPERPARAMS_DICT['val_SNR_end'] = params
            HYPERPARAMS_DICT['train_SNR_start'] = params
            HYPERPARAMS_DICT['train_SNR_end'] = params
        elif plot_type == 'plot_non_periodic':
            HYPERPARAMS_DICT['val_SNR_start'] = params
            HYPERPARAMS_DICT['val_SNR_end'] = params
            HYPERPARAMS_DICT['train_SNR_start'] = params
            HYPERPARAMS_DICT['train_SNR_end'] = params
            HYPERPARAMS_DICT['val_frames'] = val_frames
            HYPERPARAMS_DICT['word_seed'] = word_seed
            HYPERPARAMS_DICT['noise_seed'] = noise_seed
        else:
            raise ValueError("No such plot type!!!")

        HYPERPARAMS_DICT['n_symbols'] = n_symbol
        HYPERPARAMS_DICT['val_block_length'] = val_block_length
        HYPERPARAMS_DICT['train_block_length'] = val_block_length
        HYPERPARAMS_DICT['fading_in_channel'] = True if channel_coefficients == 'time_decay' else False
        HYPERPARAMS_DICT['channel_coefficients'] = channel_coefficients
        HYPERPARAMS_DICT['channel_type'] = channel_type

        if plot_type == 'plot_by_SNR' or plot_type == 'plot_non_periodic':
            current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + \
                             str(HYPERPARAMS_DICT['channel_type']) + '_' + \
                             str(HYPERPARAMS_DICT['val_SNR_start']) + '_' + \
                             str(HYPERPARAMS_DICT['val_block_length']) + '_' + \
                             str(HYPERPARAMS_DICT['n_symbols']) + '_' + \
                             str(HYPERPARAMS_DICT['meta_subframes'])
            add_viterbinet(all_curves, current_params, trial_num=1)
            add_onlinemetaviterbinet(all_curves, current_params, trial_num=1)
        elif plot_type == 'plot_by_meta_frames':
            current_params1 = HYPERPARAMS_DICT['channel_coefficients'] + '_' + \
                              str(HYPERPARAMS_DICT['channel_type']) + '_' + \
                              str(HYPERPARAMS_DICT['val_SNR_start']) + '_' + \
                              str(HYPERPARAMS_DICT['val_block_length']) + '_' + \
                              str(HYPERPARAMS_DICT['n_symbols'])
            current_params2 = current_params1 + '_' + str(HYPERPARAMS_DICT['meta_subframes'])
            add_viterbinet(all_curves, current_params1, trial_num=3)
            add_onlinemetaviterbinet(all_curves, current_params2, trial_num=3)
        elif plot_type == 'plot_by_self_supervised_iterations':
            HYPERPARAMS_DICT['self_supervised_iterations'] = 200
            current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + \
                             str(HYPERPARAMS_DICT['channel_type']) + '_' + \
                             str(HYPERPARAMS_DICT['val_SNR_start']) + '_' + \
                             str(HYPERPARAMS_DICT['val_block_length']) + '_' + \
                             str(HYPERPARAMS_DICT['n_symbols'])
            add_viterbinet(all_curves, current_params, trial_num=3)
            HYPERPARAMS_DICT['self_supervised_iterations'] = 50
            current_params2 = current_params + '_' + str(HYPERPARAMS_DICT['self_supervised_iterations'])
            add_onlinemetaviterbinet(all_curves, current_params2, trial_num=3)
        else:
            raise ValueError("No such plot type!!!")

        # add_joint_viterbinet(all_curves, current_params)
        # add_mismatched_joint_viterbinet(all_curves, current_params)
        # add_joint_rnn(all_curves, current_params)
        # add_mismatched_viterbinet(all_curves, current_params)
        # add_rnn(all_curves, current_params)
        # add_mismatched_onlinemetaviterbinet(all_curves, current_params, trial_num=1)
        # add_online_metarnn(all_curves, current_params)
        # add_viterbi(all_curves, current_params)

        if plot_by_block:
            plot_all_curves_aggregated(all_curves, val_block_length, n_symbol, snr)

    if not plot_by_block:
        plot_schematic(all_curves, parameters, xlabel)
