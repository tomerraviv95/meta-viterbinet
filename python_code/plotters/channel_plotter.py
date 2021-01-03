from dir_definitions import COST2100_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

memory_length = 4
channel_coefficients = 'cost2100'  # 'time_decay','cost2100'
if channel_coefficients == 'cost2100':
    total_h = np.empty([200, memory_length])
    for i in range(memory_length):
        total_h[:, i] = scipy.io.loadmat(os.path.join(COST2100_DIR, f'h_0.0{2 * i + 1}'))[
            'h_channel_response_mag'].reshape(-1)
    total_h = (total_h - total_h.min()) / (total_h.max() - total_h.min())
elif channel_coefficients == 'time_decay':
    gamma = 0.2
    total_h = np.empty([200, memory_length])
    for index in range(200):
        h = np.reshape(np.exp(-gamma * np.arange(memory_length)), [1, memory_length])
        fading_taps = np.array([51, 39, 33, 21])
        h *= (0.8 + 0.2 * np.cos(2 * np.pi * index / fading_taps)).reshape(1, memory_length)
        total_h[index] = h
else:
    raise ValueError("No such channel coefficients!!!")
for i in range(memory_length):
    plt.plot(total_h[:, i], label=f'Tap {i}')
plt.title('Channel Magnitude versus Block Index')
plt.xlabel('Block Index')
plt.ylabel('Magnitude')
plt.legend()
plt.show()
