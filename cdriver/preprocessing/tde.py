import numpy as np
try:
    import torch
except:
    print('torch not imported, this may lead to errors')


# def time_delay_embedding(data, delay, dimension):
#     """
#     Creates a time delay embedding of a time series data with a specified delay and dimension.
#
#     Parameters:
#         data (1D array): The time series data
#         delay (int): The delay to use in the embedding
#         dimension (int): The dimension of the embedding
#
#     Returns:
#         (2D array): The time delay embedding
#     """
#     embedding = np.zeros((len(data) - (dimension - 1) * delay, dimension))
#     for i in range(dimension):
#         embedding[:, i] = data[i * delay: i * delay + embedding.shape[0]]
#     return embedding

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


def cropper(x, n, location):
    """Crops the first or last n elements from a dataset along the first axis

    :param arraylike x: The dataset to crop
    :param n: The number of elements to crop, if n<0, it does nothing
    :param str location: location to crop from. Either 'first' or 'last'
    :return: The cropped dataset
    """
    if n>0:
        if location == 'first':
            return x[n:]
        elif location == 'last':
            return x[:-n]
        else:
            raise ValueError('location must be either "first" or "last"')
    else:
        return x
class TimeDelayEmbeddingTransform:
    def __init__(self, embedding_dim, delay):
        self.embedding_dim = embedding_dim
        self.delay = delay

    def __call__(self, x):
        """
        :param x: Input time series tensor of shape (sequence_length,)
        :return: Embedded time series tensor of shape (sequence_length - (embedding_dim - 1) * delay, embedding_dim)
        """
        seq_len = x.shape[0]
        embedded_seq_len = seq_len - (self.embedding_dim - 1) * self.delay

        embedded_series = torch.zeros(embedded_seq_len, self.embedding_dim)
        for j in range(self.embedding_dim):
            embedded_series[:, j] = x[j * self.delay : j * self.delay + embedded_seq_len].squeeze()

        return embedded_series