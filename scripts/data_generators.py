'''Helper functions'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt

def comp_ccorr(y, y_recon):
  """Computes crosscorrelation
  """
  T = len(y)
  c = np.correlate(1/T * scale(y[:T]), scale(y_recon), mode='full')
  tau = np.arange(-T+1, T)
  return tau, c

def get_maxes(tau, c):
  """Gets the argmax and max of |xcorr|
  """
  return tau[np.argmax(np.abs(c))], max(np.abs(c))



def time_delay_embedding(series, delay=1, dimension=3):
    """
    Perform time delay embedding of a time series.

    Parameters:
        series (numpy.ndarray): 1-dimensional array representing the time series data.
        delay (int): The time delay between consecutive samples.
        dimension (int): The number of dimensions (embedding dimension).

    Returns:
        numpy.ndarray: The embedded time series with shape (N, dimension), where N is the number
                       of embedded points.
    """
    num_samples = len(series) - (dimension - 1) * delay
    embedded_series = np.zeros((num_samples, dimension))
    for i in range(num_samples):
        embedded_series[i] = series[i:i + dimension * delay:delay]
    return embedded_series

def save_results(fname, r, N, method, dataset):
    """Saves results to a file and returns the dataframe
    """
    df = pd.DataFrame(np.array([range(N), r, N * [method], N * [dataset]]).T,
                             columns=['data_id', 'r', 'method', 'dataset'])
    df.to_csv(fname)
    return df

def train_test_split(X, Y, z, train_size=0.8):
    """Splits data into train and test sets
    """
    N = len(X)
    X_train, Y_train, z_train = X[:int(train_size*N)], Y[:int(train_size*N)], z[:int(train_size*N)]
    X_test, Y_test, z_test = X[int(train_size*N):], Y[int(train_size*N):], z[int(train_size*N):]
    return X_train, Y_train, z_train, X_test, Y_test, z_test

def train_valid_test_split(X, Y, z, train_size=0.8, valid_size=0.1):
    """
    Splits data into train, validation and test sets
    """
    N = len(X)
    X_train, Y_train, z_train = (X[:int(train_size * N)],
                                 Y[:int(train_size * N)],
                                 z[:int(train_size * N)])
    X_valid, Y_valid, z_valid = (X[int(train_size * N):int((train_size + valid_size) * N)],
                                 Y[int(train_size * N):int((train_size + valid_size) * N)],
                                 z[int(train_size * N):int((train_size + valid_size) * N)])
    X_test, Y_test, z_test = (X[int((train_size + valid_size) * N):],
                              Y[int((train_size + valid_size) * N):],
                              z[int((train_size + valid_size) * N):])
    return X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test

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