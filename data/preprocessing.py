import numpy as np
from scipy.signal import butter, filtfilt


def apply_car(data):
    """Common Average Referencing (Equation 1)"""
    return data - np.mean(data, axis=1, keepdims=True)


def bandpass_filter(data, fs=250, lowcut=8, highcut=30, order=5):
    """Zero-phase Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)
