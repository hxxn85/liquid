from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

def monopuls(t, tp):
    a = 1/(2*tp/10)
    ys = -2*(a**2)*t*np.exp(-(a**2)*t**2)
    return ys / np.max(ys)

def mag1db(x):
    return 10*np.log10(np.array(x).clip(1e-14))

fs = 100e9
ts = 1/fs
tp = 2000e-12
n = 512
t = np.arange(n)*ts - n/2*ts

#%%
x = monopuls(t, 250e-12)
plt.plot(t*1e9, x)
plt.xlim([-2, 2])
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.grid(alpha=0.5)
plt.show()

f, pxx = signal.periodogram(x, fs, 'hann', nfft=2048)
plt.plot(f/1e9, mag1db(pxx)+55)
plt.xlim([0, 20])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Voltage (dBmv/MHz)')
plt.grid(alpha=0.5)
plt.hlines(np.max(mag1db(pxx))+45, 0, 20, 'r', linestyles='--')
plt.show()