import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
from sp import *

import dataset

param = dataset.Param()
x = dataset.Dataset().rx(8)
target = ['toluene', 'water', 'space']
y = {lqd: pulseint(x[lqd], 'coherent') for lqd in target}

plt.subplot(3,1,1)
plt.plot(param.t, y['water'])

r = pd.read_csv('waveform/rigid.csv').ys.values[490:522]
plt.subplot(3,1,2)
plt.plot(param.t, y['water'] - y['space'])

plt.subplot(3,1,3)
rxy = signal.correlate(y['water'] - y['space'], r, mode='same')
plt.plot(param.t, rxy)
plt.tight_layout()
plt.show()

def colormesh(x, ax):
    f, t, sxx = signal.spectrogram(x, 42e9, nperseg=14, noverlap=13, nfft=512)
    ax.pcolormesh(t * 1e9, f * 1e-9, sxx, shading='gouraud', cmap='jet')
    ax.set_ylim([0, 10])
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_xlabel('$time (ns)$')
    ax.set_ylabel('$frequency (GHz)$')

fig = plt.figure()
ax = [fig.add_subplot(3,1,i+1) for i in range(3)]
colormesh(y['water'], ax[0])
colormesh(y['water'] - y['space'], ax[1])
colormesh(rxy, ax[2])
plt.tight_layout()
plt.show()
