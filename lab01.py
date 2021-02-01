import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sp

#%% load water data
water = pd.read_csv('dataset/water.csv')
slave = [water.groupby('Unnamed: 0').get_group(f'Slave{i+1}') for i in range(10)]

data = np.array([slave[i].mean() for i in range(len(slave))])
#%% todo: normalize and pulse integration

#%% todo: singluar value decomposition for clutter removal

#%% plot data
n = 512
fs = 42e9
ts = 1/fs
t = np.linspace(0, n, n, endpoint=False)*ts
x = signal.gausspulse(t-n/2*ts, fc=5e9, bw=0.5)
y = (data[0] - np.mean(data[0])) / np.std(data[0])
plt.plot(t, x, t, y)
plt.grid()
plt.show()

#%%
rxy, lags = sp.xcorr(x, y)
plt.plot(lags[512:]*ts, sp.mag2db(rxy[512:]))
plt.grid()
plt.ylim([-30, 100])
plt.show()

#%%
# start = lags[np.argmax(rxy)]
# plt.plot(y[start:start+12])
# plt.plot(x)
# plt.show()

#%%
f, t, sxx = signal.spectrogram(y[0:], fs, nperseg=14, noverlap=13, nfft=512)
plt.pcolormesh(t*1e9, f*1e-9, sxx, shading='gouraud', cmap='jet')
plt.ylim([0, 10])
plt.grid(linestyle='--', alpha=0.5)
plt.xlabel('$time (ns)$')
plt.ylabel('$frequency (GHz)$')
plt.show()