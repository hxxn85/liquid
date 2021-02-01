from scipy import signal
import numpy as np

def xcorr(x, y, mode='full', scaleopt='None'):
    rxy = signal.correlate(x, y, mode=mode)
    lags = signal.correlation_lags(len(x), len(y), mode=mode)
    if scaleopt == 'normalized':
        rxx = signal.correlate(x, x, mode='valid')
        ryy = signal.correlate(y, y, mode='valid')
        rxy /= np.sqrt(rxx*ryy)

    return rxy, lags

def mag1db(x):
    return 10 * np.log10(np.abs(np.array(x).clip(1e-14)))

def mag2db(x):
    return 20*np.log10(np.abs(np.array(x).clip(1e-14)))

def pulseint(x, method='noncoherent'):
    x = np.asarray(x)
    if method == 'noncoherent':
        return np.sqrt(np.sum(np.power(np.absolute(x),2),0))
    elif method == 'coherent':
        return np.sum(x, 0)
    else:
        raise ValueError("Acceptable method flags are 'coherent', or 'noncoherent'.")