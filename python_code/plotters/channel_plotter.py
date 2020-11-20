from dir_definitions import COST2100_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

memory_length = 4
total_h = np.empty([100, memory_length])
for i in range(memory_length):
    total_h[:, i] = scipy.io.loadmat(os.path.join(COST2100_DIR, f'h_{0.02 * (i + 1)}'))[
        'h_channel_response_mag'].reshape(-1)
# scale min-max values of h to the range 0-1
total_h = (total_h - total_h.min()) / (total_h.max() - total_h.min())
for i in range(memory_length):
    plt.plot(total_h[:, i], label=f'Tap {i + 1}')
plt.title('Channel Magnitude versus Block Index')
plt.xlabel('Block Index')
plt.ylabel('Magnitude')
plt.legend()
plt.show()
