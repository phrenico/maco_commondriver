import numpy as np


def shuffle_phase(x):
    """shuffles the phase of the signal in Fourier domain

    :param x: signal
    :param sf: sampling rate
    :return: fourier-shuffled signal
    """
    X = np.fft.fft(x)
    phase = np.pi * ( 2 * np.random.rand(len(X)) - 1)
    X_shuffled = np.abs(X) * np.exp(1j * phase)
    x_shuffled = np.fft.ifft(X_shuffled)
    return np.real(x_shuffled)