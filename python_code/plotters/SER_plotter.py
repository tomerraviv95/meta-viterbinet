import datetime
import os

from python_code.trainers.VA.va_trainer import VATrainer
from dir_definitions import FIGURES_DIR
import matplotlib.pyplot as plt
import matplotlib as mpl

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

MARKERS_DICT = {'VA': 'o'}
COLORS_DICT = {'VA': '#0F9D58'}  # google green

dec = VATrainer()
ber_total, fer_total = dec.evaluate()
snr_range = dec.snr_range['val']
method_name = dec.get_name()

current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}_{dec.block_length}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))
plt.figure()

plt.plot(snr_range, ber_total, label=method_name, marker=MARKERS_DICT[method_name], color=COLORS_DICT[method_name],
         linewidth=2.2, markersize=12)
plt.yscale('log')
plt.ylabel('SER')
plt.xlabel('$E_b/N_0$ [dB]')
plt.grid(which='both', ls='--')
plt.xlim([snr_range[0] - 0.1, snr_range[-1] + 0.1])
plt.legend(loc='lower left', prop={'size': 15})

plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER.png'), bbox_inches='tight')

plt.show()
