import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

#TODO: remove common pattern from the signal using correlation

def str2float(x):
    return [float(k) for k in x[1:]]


fs = 42e9
t = np.arange(512) / fs
f = open('dataset/water.csv', 'r', encoding='utf-8')
reader = csv.reader(f)
data = list(reader)
rx = [np.array(x) for x in data if 'Slave2' in x[0]] 
rx = [str2float(x) for x in rx]

rx = rx - np.mean(rx)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5))
ax1.plot(t, rx.T)
f, Pxx = periodogram(rx, fs=42e9)
ax2.plot(f, Pxx.T)
f, Pxx = welch(rx, fs=42e9, nperseg=128, noverlap=64, nfft=512,
               return_onesided=True, scaling='density')
ax3.plot(f, Pxx.T)
plt.show()

print(f'peak frequency: {f[np.argmax(Pxx[0])]} Hz')
