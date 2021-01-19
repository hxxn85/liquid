import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg

import dataset

def pulseint(x, method='noncoherent'):
    if method not in ['noncoherent', 'coherent']:
        raise ValueError("method should be 'coherent' or 'noncoherent'")

    if method == 'noncoherent':
        return np.sqrt(np.sum(np.power(np.absolute(x), 2), 0))
    elif method == 'coherent':
        return np.sum(x)

param = dataset.Param()
x = dataset.Dataset().rx(8)
target = ['toluene', 'water', 'space']
y = {lqd: pulseint(x[lqd], 'coherent').values for lqd in target}

plt.figure()
plt.plot(param.t*1e9, y['toluene'], alpha=.7, label='toluene')
plt.plot(param.t*1e9, y['water'], alpha=.7, label='water')
plt.plot(param.t*1e9, y['space'], alpha=.7, label='space')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()

def colormesh(x, ax):
    f, t, sxx = signal.spectrogram(x, 42e9, nperseg=14, noverlap=13, nfft=512)
    ax.pcolormesh(t * 1e9, f * 1e-9, sxx, shading='gouraud', cmap='jet')
    ax.set_ylim([0, 10])
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_xlabel('$time (ns)$')
    ax.set_ylabel('$frequency (GHz)$')

fig = plt.figure()
ax =  [fig.add_subplot(len(target),1,i+1) for i in range(len(target))]
[colormesh(y[x], ax, x) for x, ax in zip(y, ax)]
plt.tight_layout()
plt.show()

#%%
y = pulseint(x['water'], 'coherent').values
z = pulseint(x['space'], 'coherent').values
A = np.array([y, z])
u, s, vt = linalg.svd(A.T)


# plt.plot(param.t*1e9, A.T)
# plt.plot(param.t*1e9, y)
# plt.show()
