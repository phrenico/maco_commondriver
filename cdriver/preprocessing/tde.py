import numpy as np


def time_delay_embedding(data, delay, dimension):
    """
    Creates a time delay embedding of a time series data with a specified delay and dimension.

    Parameters:
        data (1D array): The time series data
        delay (int): The delay to use in the embedding
        dimension (int): The dimension of the embedding

    Returns:
        (2D array): The time delay embedding
    """
    embedding = np.zeros((len(data) - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedding[:, i] = data[i * delay: i * delay + embedding.shape[0]]
    return embedding