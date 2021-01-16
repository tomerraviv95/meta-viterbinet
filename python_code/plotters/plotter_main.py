from dir_definitions import WEIGHTS_DIR
import os
from python_code.plotters.plotter_utils import get_ser_plot, plot_all_curves_aggregated
from python_code.trainers.META_LSTM.meta_lstm_trainer import MetaLSTMTrainer
from python_code.trainers.META_VNET.metavnet_trainer import METAVNETTrainer
from python_code.trainers.LSTM.lstm_trainer import LSTMTrainer
from python_code.trainers.VA.va_trainer import VATrainer
from python_code.trainers.VNET.vnet_trainer import VNETTrainer
from python_code.plotters.plotter_config import *



def add_viterbi(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = VATrainer(val_SNR_start=snr,
                    val_SNR_end=snr,
                    train_SNR_start=snr,
                    train_SNR_end=snr,
                    val_SNR_step=2,
                    train_SNR_step=2,
                    val_block_length=val_block_length,
                    train_block_length=val_block_length,
                    noisy_est_var=0,
                    fading_in_channel=False,
                    fading_in_decoder=False,
                    use_ecc=True,
                    gamma_start=0.2,
                    gamma_end=0.2,
                    gamma_num=1,
                    channel_type='ISI_AWGN',
                    channel_coefficients='cost2100',
                    subframes_in_frame=25,
                    val_frames=VAL_FRAMES,
                    eval_mode='by_word',
                    n_symbols=n_symbol,
                    fading_taps_type=1,
                    self_supervised=False,
                    online_meta=False,
                    weights_dir=os.path.join(WEIGHTS_DIR,
                                             f'training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'Viterbi - Full CSI'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_viterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = VNETTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=False,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='cost2100',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=1,
                      self_supervised=False,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'Joint ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_joint_rnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = LSTMTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=False,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='cost2100',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=1,
                      self_supervised=False,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'JointRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_viterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = VNETTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=False,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='cost2100',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=1,
                      self_supervised=True,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'ViterbiNet'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_rnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = LSTMTrainer(val_SNR_start=snr,
                      val_SNR_end=snr,
                      train_SNR_start=snr,
                      train_SNR_end=snr,
                      val_SNR_step=2,
                      train_SNR_step=2,
                      val_block_length=val_block_length,
                      train_block_length=val_block_length,
                      noisy_est_var=0,
                      fading_in_channel=False,
                      fading_in_decoder=False,
                      use_ecc=True,
                      gamma_start=0.2,
                      gamma_end=0.2,
                      gamma_num=1,
                      channel_type='ISI_AWGN',
                      channel_coefficients='cost2100',
                      subframes_in_frame=25,
                      val_frames=VAL_FRAMES,
                      eval_mode='by_word',
                      n_symbols=n_symbol,
                      fading_taps_type=1,
                      self_supervised=True,
                      online_meta=False,
                      weights_dir=os.path.join(WEIGHTS_DIR,
                                               f'rnn_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'LSTM'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


def add_onlinemetaviterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = METAVNETTrainer(val_SNR_start=snr,
                          val_SNR_end=snr,
                          train_SNR_start=snr,
                          train_SNR_end=snr,
                          val_SNR_step=2,
                          train_SNR_step=2,
                          val_block_length=val_block_length,
                          train_block_length=val_block_length,
                          noisy_est_var=0,
                          fading_in_channel=False,
                          fading_in_decoder=False,
                          use_ecc=True,
                          gamma_start=0.2,
                          gamma_end=0.2,
                          gamma_num=1,
                          channel_type='ISI_AWGN',
                          channel_coefficients='cost2100',
                          subframes_in_frame=25,
                          val_frames=VAL_FRAMES,
                          eval_mode='by_word',
                          n_symbols=n_symbol,
                          fading_taps_type=1,
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


def add_online_metarnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients):
    dec = MetaLSTMTrainer(val_SNR_start=snr,
                          val_SNR_end=snr,
                          train_SNR_start=snr,
                          train_SNR_end=snr,
                          val_SNR_step=2,
                          train_SNR_step=2,
                          val_block_length=val_block_length,
                          train_block_length=val_block_length,
                          noisy_est_var=0,
                          fading_in_channel=False,
                          fading_in_decoder=False,
                          use_ecc=True,
                          gamma_start=0.2,
                          gamma_end=0.2,
                          gamma_num=1,
                          channel_type='ISI_AWGN',
                          channel_coefficients='cost2100',
                          subframes_in_frame=25,
                          val_frames=VAL_FRAMES,
                          eval_mode='by_word',
                          n_symbols=n_symbol,
                          fading_taps_type=1,
                          self_supervised_iterations=200,
                          self_supervised=True,
                          ser_thresh=0.02,
                          online_meta=False,
                          buffer_empty=True,
                          weights_init='last_frame',
                          weights_dir=os.path.join(WEIGHTS_DIR,
                                                   f'rnn_meta_training_{val_block_length}_{n_symbol}_channel1'))
    method_name = f'OnlineRNN'
    ser = get_ser_plot(dec, run_over=run_over,
                       method_name=method_name + f' - SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}')
    all_curves.append((ser, method_name, val_block_length, n_symbol))


VAL_FRAMES = 20

if __name__ == '__main__':
    run_over = False
    parameters = [(7, 120)]  # ,(8, 120)
    n_symbol = 2
    channel_coefficients = 'cost2100'  # 'time_decay','cost2100'

    for snr, val_block_length in parameters:
        all_curves = []
        print(snr, val_block_length, n_symbol)
        add_joint_viterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_joint_rnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_viterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_rnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_onlinemetaviterbinet(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_online_metarnn(all_curves, val_block_length, n_symbol, snr, channel_coefficients)
        add_viterbi(all_curves, val_block_length, n_symbol, snr, channel_coefficients)

        plot_all_curves_aggregated(all_curves, val_block_length, n_symbol, snr)
        # plot_all_curves(all_curves, val_block_length, n_symbol)
    # plot_schematic(all_curves, snr_values)
