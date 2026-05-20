import numpy as np


def shuffle_phase(x):
    """Shuffles the phase of the signal in Fourier domain.

    :param x: 1-D numpy array, the input signal
    :return: Fourier-phase-shuffled signal (real part only)
    """
    X = np.fft.fft(x)
    phase = np.pi * ( 2 * np.random.rand(len(X)) - 1)
    X_shuffled = np.abs(X) * np.exp(1j * phase)
    x_shuffled = np.fft.ifft(X_shuffled)
    return np.real(x_shuffled)