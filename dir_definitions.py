import os
# main folders
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
# subfolders
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
COST2100_DIR = os.path.join(RESOURCES_DIR, 'cost2100_channel')
NON_PERIODIC_PATH = os.path.join(RESOURCES_DIR, 'non_periodic_channel_taps')
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
