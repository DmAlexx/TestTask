# utils/data_loader.py

import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist(normalize=True, reshape=True):
    """
    Loads the MNIST dataset and optionally normalizes and reshapes it.
    
    :param normalize: Whether to normalize pixel values to the range [0, 1].
    :param reshape: Whether to reshape data for CNNs (28x28x1).
    :return: Tuple (X_train, y_train, X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if normalize:
        X_train, X_test = X_train / 255.0, X_test / 255.0

    if reshape:
        X_train = X_train.reshape(-1, 28, 28, 1)  # Reshape for CNN input
        X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test
