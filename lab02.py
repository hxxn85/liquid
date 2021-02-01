import matplotlib.pyplot as plt
from scipy import linalg
from sp import *

import dataset

param = dataset.Param()
x = dataset.Dataset().rx(10)
target = ['toluene', 'water', 'space']
y = {lqd: pulseint(x[lqd], 'coherent') for lqd in target}

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
ax = [fig.add_subplot(len(target),1,i+1) for i in range(len(target))]
[colormesh(y[x], ax) for x, ax in zip(y, ax)]
plt.tight_layout()
plt.show()

#%% background subtraction
water = pulseint(x['bottle'], 'coherent')
background = pulseint(x['space'], 'coherent')

r = water - background
f, t, srr = signal.spectrogram(r, 42e9, nperseg=14, noverlap=13, nfft=512)
fig = plt.figure()
ax = [fig.add_subplot(2,1,i) for i in [1, 2]]
ax[0].plot(param.t*1e9, r)
ax[1].pcolormesh(t*1e9, f/1e9, srr, shading='gouraud', cmap='jet')
plt.show()

#%%
A = x['water'].T
u, s, vt = linalg.svd(A.T, full_matrices=False)
A2 = u[:, 2:] @ np.diag(s[2:]) @ vt[2:, :]
r = pulseint(A2, 'coherent')
# plt.plot(param.t*1e9, r)
# plt.show()

f, t, sxx = signal.spectrogram(r, 42e9, nperseg=14, noverlap=13, nfft=512)
plt.pcolormesh(t * 1e9, f * 1e-9, sxx, shading='gouraud', cmap='jet')
plt.grid(linestyle='--', alpha=0.5)
plt.ylim([0, 10])
plt.xlabel('$time (ns)$')
plt.ylabel('$frequency (GHz)$')
plt.show()

#%%
ref = signal.gausspulse(param.t-param.n/2*param.ts, fc=5e9, bw=0.5)
rxy, lags = xcorr(ref, r)
plt.plot(lags[512:]*param.ts*1e9, mag2db(rxy[512:]))
plt.grid()
plt.ylim([-30, 100])
plt.show()