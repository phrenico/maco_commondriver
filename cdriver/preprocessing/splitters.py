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